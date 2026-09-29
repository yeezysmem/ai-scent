from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from PIL import Image

from scent_ai.config import CARTRIDGES, SCENT_SOURCES, SOURCE_TO_CARTRIDGE
from scent_ai.smell_engine import classify_source, clamp


class RuntimeSmellEngine:
    """
    Runtime-шар між AI-моделями та фізичним пристроєм.

    Функції:
    - аналізує новий кадр через YOLO + CLIP;
    - згладжує інтенсивності (EMA);
    - обмежує кількість одночасно активних картриджів (Top-N);
    - фільтрує незначні зміни та керує інтервалом відправки.
    """

    def __init__(
        self,
        segmenter: Any = None,
        classifier: Any = None,
        detector: Any = None,
        activation_threshold: float = 0.10,
        smoothing_alpha: float = 0.35,
        change_threshold: float = 0.05,
        update_interval: float = 0.5,
        max_active_cartridges: int = 3,
    ) -> None:
        if not 0.0 <= activation_threshold <= 1.0:
            raise ValueError("activation_threshold must be between 0.0 and 1.0.")

        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be greater than 0 and at most 1.")

        if not 0.0 <= change_threshold <= 1.0:
            raise ValueError("change_threshold must be between 0.0 and 1.0.")

        if update_interval < 0.0:
            raise ValueError("update_interval cannot be negative.")

        if max_active_cartridges < 1:
            raise ValueError("max_active_cartridges must be at least 1.")

        self.segmenter = segmenter
        self.classifier = classifier
        self.detector = detector

        self.activation_threshold = activation_threshold
        self.smoothing_alpha = smoothing_alpha
        self.change_threshold = change_threshold
        self.update_interval = update_interval
        self.max_active_cartridges = max_active_cartridges

        self.current_state: dict[str, float] = {cartridge: 0.0 for cartridge in CARTRIDGES}
        self.last_sent_state: dict[str, float] = {cartridge: 0.0 for cartridge in CARTRIDGES}
        self.last_analysis: dict[str, Any] | None = None
        self.last_update_time: float = 0.0

    def process(self, image: Image.Image, force: bool = False) -> dict[str, Any]:
        """Основний метод обробки кадру в реальному часі."""
        return self.process_with_yolo_clip(image=image, force=force)

    def process_with_yolo_clip(self, image: Image.Image, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        
        # Перевірка інтервалу оновлення (якщо не примусово)
        if not force and (now - self.last_update_time < self.update_interval):
            return {
                "cartridges": self.current_state,
                "analysis": self.last_analysis or {},
                "should_send": False,
            }

        raw_cartridges = {c: 0.0 for c in CARTRIDGES}

        # 1. Прохід YOLO-World
        yolo_results = {}
        if self.detector:
            try:
                yolo_results = self.detector.detect(image)
                yolo_weights = self.detector.get_cartridge_weights(image)
                for cartridge, weight in yolo_weights.items():
                    if cartridge in raw_cartridges:
                        raw_cartridges[cartridge] += weight
            except Exception as e:
                print(f"⚠️ Помилка YOLO detector: {e}")

        # 2. Глобальний прохід CLIP
       # 2. Глобальний прохід CLIP
        final_sources: dict[str, float] = {}
        if self.classifier:
            for source_name, source_config in SCENT_SOURCES.items():
                try:
                    res = classify_source(image, self.classifier, source_config)
                    
                    # Безпечне витягування значення score з результату
                    score = 0.0
                    if isinstance(res, list) and len(res) > 0:
                        # Якщо повернувся список dict'ів [{'label': ..., 'score': 0.85}]
                        first_item = res[0]
                        if isinstance(first_item, dict):
                            score = float(first_item.get("score", 0.0))
                    elif isinstance(res, dict):
                        # Якщо повернувся словник
                        score = float(res.get("score", res.get(source_name, 0.0)))

                    final_sources[source_name] = score

                    # Мапимо джерела на картриджі
                    mapping = SOURCE_TO_CARTRIDGE.get(source_name, {})
                    for cartridge, weight in mapping.items():
                        if cartridge in raw_cartridges:
                            raw_cartridges[cartridge] += score * weight

                except Exception as e:
                    print(f"⚠️ Помилка CLIP для джерела {source_name}: {e}")

        # 3. Обмеження діапазону (Clamp [0.0, 1.0])
        clamped_cartridges = {c: clamp(val) for c, val in raw_cartridges.items()}

        # 4. Згладжування (Exponential Moving Average)
        smoothed_state = self._smooth(self.current_state, clamped_cartridges)

        # 5. Топ-N найсильніших картриджів
        active_state = self._apply_max_active_cartridges(smoothed_state)

        # 6. Перевірка на значні зміни
        should_send = force or self._has_significant_change(active_state, self.last_sent_state)

        # Оновлення внутрішніх станів
        self.current_state = active_state
        self.last_analysis = {
            "yolo_results": yolo_results,
            "final_sources": final_sources,
        }

        if should_send:
            self.last_sent_state = active_state.copy()
            self.last_update_time = now

        return {
            "cartridges": self.current_state,
            "analysis": self.last_analysis,
            "should_send": should_send,
        }

    def _apply_max_active_cartridges(self, state: dict[str, float]) -> dict[str, float]:
        """Залишає активними тільки N найсильніших картриджів, інші обнуляє."""
        sorted_cartridges = sorted(state.items(), key=lambda x: x[1], reverse=True)
        active_keys = {k for k, v in sorted_cartridges[: self.max_active_cartridges] if v > 0}
        return {k: (v if k in active_keys else 0.0) for k, v in state.items()}

    def _smooth(
        self,
        previous: Mapping[str, float],
        incoming: Mapping[str, float],
    ) -> dict[str, float]:
        """Exponential moving average з порогом активації."""
        result: dict[str, float] = {}

        for cartridge in CARTRIDGES:
            old_value = float(previous.get(cartridge, 0.0))
            new_value = float(incoming.get(cartridge, 0.0))

            smoothed_value = (
                self.smoothing_alpha * new_value
                + (1.0 - self.smoothing_alpha) * old_value
            )

            if smoothed_value < self.activation_threshold:
                smoothed_value = 0.0

            result[cartridge] = clamp(smoothed_value)

        return result

    def _has_significant_change(
        self,
        current: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> bool:
        """Перевіряє, чи перевищує зміна поріг change_threshold."""
        return any(
            abs(float(current.get(cartridge, 0.0)) - float(previous.get(cartridge, 0.0)))
            >= self.change_threshold
            for cartridge in CARTRIDGES
        )

    def get_state(self) -> dict[str, float]:
        """Повертає поточний стан картриджів."""
        return self.current_state.copy()

    def get_percent_state(self) -> dict[str, int]:
        """Перетворює 0.0–1.0 у 0–100 для UI."""
        return {
            cartridge: round(clamp(value) * 100)
            for cartridge, value in self.current_state.items()
        }

    def get_pwm_state(self, max_pwm: int = 255) -> dict[str, int]:
        """Перетворює 0.0–1.0 у PWM-значення для ESP32."""
        if max_pwm < 1:
            raise ValueError("max_pwm must be at least 1.")

        return {
            cartridge: round(clamp(value) * max_pwm)
            for cartridge, value in self.current_state.items()
        }

    def stop(self) -> dict[str, float]:
        """Примусово вимикає всі картриджі."""
        stopped = {cartridge: 0.0 for cartridge in CARTRIDGES}
        self.current_state = stopped.copy()
        self.last_sent_state = stopped.copy()
        self.last_update_time = time.monotonic()
        return stopped

    def reset(self) -> None:
        """Скидає стан двигуна."""
        self.current_state = {cartridge: 0.0 for cartridge in CARTRIDGES}
        self.last_sent_state = {cartridge: 0.0 for cartridge in CARTRIDGES}
        self.last_analysis = None
        self.last_update_time = 0.0