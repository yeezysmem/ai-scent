from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mss
from PIL import Image


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture:
    def __init__(
        self,
        monitor_index: int = 1,
        region: CaptureRegion | None = None,
        window_manager: Any | None = None,
    ) -> None:
        if monitor_index < 1:
            raise ValueError("monitor_index must be 1 or greater.")

        self.monitor_index = monitor_index
        self.region = region
        self.window_manager = window_manager
        self._sct = mss.mss()

    def capture(self, window_id: int | None = None) -> Image.Image:
        """
        Захоплює обране вікно (через native macOS screencapture) 
        або весь екран (через mss).
        """
        # 1. Якщо задано window_id, пробуємо native macOS захоплення вікна
        if window_id is not None and window_id > 0:
            try:
                img = self._capture_mac_window(window_id)
                if img is not None:
                    return img
            except Exception as e:
                print(f"⚠️ Native macOS capture failed for window {window_id}: {e}")

        # 2. Якщо window_id не задано або native capture не спрацював — захоплюємо весь екран/область
        target = self._get_default_target()
        screenshot = self._sct.grab(target)

        return Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.rgb,
        )

    def _capture_mac_window(self, window_id: int) -> Image.Image | None:
        """Використовує системну утиліту macOS screencapture для ізольованого знімка вікна."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # -l<id> захоплює тільки вікно за його CGWindowNumber
            # -x вимикає звук затвора камери
            cmd = ["screencapture", f"-l{window_id}", "-x", str(tmp_path)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

            if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                img = Image.open(tmp_path).convert("RGB")
                return img
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return None

    def _get_default_target(self) -> dict[str, int]:
        if self.region is not None:
            return self.region.as_mss_dict()
        return self._sct.monitors[self.monitor_index]

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()