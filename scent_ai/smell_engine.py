from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from scent_ai.config import (
    CARTRIDGES,
    SCENT_SOURCES,
    SOURCE_TO_CARTRIDGE,
)

SourceResult = dict[str, float | str]
ScoreMapping = Mapping[str, float]


def safe_float(value: Any) -> float:
    """Гарантовано перетворює будь-яке значення у звичайний Python float."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.item())
    return float(arr.mean())


def clamp(value: float) -> float:
    """Обмежує числове значення діапазоном 0.0–1.0."""
    return max(0.0, min(float(value), 1.0))


def classify_source(
    image: Image.Image,
    classifier: Any,
    config: Mapping[str, list[str]],
) -> SourceResult:
    """Визначає інтенсивність одного джерела запаху через CLIP."""
    if not isinstance(image, Image.Image):
        raise TypeError("classify_source() expects a PIL.Image.Image.")

    positive_labels = list(config.get("positive", []))
    negative_labels = list(config.get("negative", []))

    if not positive_labels or not negative_labels:
        raise ValueError("Source configuration must contain both positive and negative labels.")

    labels = positive_labels + negative_labels

    results = classifier(
        image,
        candidate_labels=labels,
    )

    scores = {
        str(item["label"]): safe_float(item["score"])
        for item in results
    }

    # Знаходимо найкращі окремі промпти для логування
    best_positive_label = max(positive_labels, key=lambda label: scores.get(label, 0.0))
    best_negative_label = max(negative_labels, key=lambda label: scores.get(label, 0.0))

    # Рахуємо сумарну ймовірнісну масу позитивної та негативної груп
    pos_sum = sum(scores.get(lbl, 0.0) for lbl in positive_labels)
    neg_sum = sum(scores.get(lbl, 0.0) for lbl in negative_labels)

    total_mass = pos_sum + neg_sum
    if total_mass > 0:
        # Відносна впевненість позитивної групи (від 0.0 до 1.0)
        pos_ratio = pos_sum / total_mass
        # Фінальний скор нормалізуємо: якщо pos_ratio > 0.5 (позитив переважає), розраховуємо шарп-коефіцієнт
        source_score = (pos_ratio - 0.5) * 2.0 if pos_ratio > 0.55 else 0.0
    else:
        source_score = 0.0

    return {
        "score": clamp(source_score),
        "positive_score": clamp(scores.get(best_positive_label, 0.0)),
        "negative_score": clamp(scores.get(best_negative_label, 0.0)),
        "positive_label": best_positive_label,
        "negative_label": best_negative_label,
    }


def get_area(object_areas: ScoreMapping, *labels: str) -> float:
    """Підсумовує площі об'єктів/сегментів."""
    total = sum(float(object_areas.get(label, 0.0)) for label in labels)
    return clamp(total)


def build_scene_areas(object_areas: ScoreMapping) -> dict[str, float]:
    """Перетворює деталізовані класи об'єктів у семантичні групи."""
    return {
        "vegetation": get_area(object_areas, "tree", "plant", "grass", "field"),
        "tree": get_area(object_areas, "tree"),
        "field": get_area(object_areas, "field"),
        "sea": get_area(object_areas, "sea", "ocean"),
        "water": get_area(object_areas, "water", "sea", "ocean", "river", "lake"),
        "ground": get_area(object_areas, "earth", "dirt", "soil", "sand", "mud"),
        "road": get_area(object_areas, "road", "sidewalk", "pavement", "street"),
        "sky": get_area(object_areas, "sky"),
        "fire_hazard": get_area(object_areas, "fire", "smoke", "flame"),
        "vehicle": get_area(object_areas, "car", "truck", "bus", "automobile"),
    }


def combine_sources_with_segmentation(
    source_scores: ScoreMapping,
    scene_areas: ScoreMapping,
) -> dict[str, float]:
    """Об'єднує оцінки CLIP із площами об'єктів."""
    combined: dict[str, float] = {}

    # 1. Хвойний ліс
    coniferous_clip = source_scores.get("coniferous_forest", 0.0)
    tree_area = scene_areas.get("tree", 0.0)
    combined["coniferous_forest"] = (
        coniferous_clip * (0.50 + 0.50 * tree_area) if scene_areas else coniferous_clip
    )

    # 2. Листяний ліс
    broadleaf_clip = source_scores.get("broadleaf_forest", 0.0)
    vegetation_area = scene_areas.get("vegetation", 0.0)
    combined["broadleaf_forest"] = (
        broadleaf_clip * (0.50 + 0.50 * vegetation_area) if scene_areas else broadleaf_clip
    )

    # 3. Океан / Море
    ocean_clip = source_scores.get("ocean_sea", 0.0)
    water_area = scene_areas.get("water", 0.0)
    combined["ocean_sea"] = (
        ocean_clip * (0.50 + 0.50 * water_area) if scene_areas else ocean_clip
    )

    # 4. Дим і вогонь
    combined["smoke_fire"] = source_scores.get("smoke_fire", 0.0)

    # 5. Ґрунт / Бруд
    soil_clip = source_scores.get("soil_mud", 0.0)
    ground_area = scene_areas.get("ground", 0.0)
    combined["soil_mud"] = (
        soil_clip * (0.50 + 0.50 * ground_area) if scene_areas else soil_clip
    )

    # 6. Дощ
    combined["rain_wetness"] = source_scores.get("rain_wetness", 0.0)

    # 7. Асфальт / Місто
    asphalt_clip = source_scores.get("asphalt_city", 0.0)
    road_area = scene_areas.get("road", 0.0)
    combined["asphalt_city"] = (
        asphalt_clip * (0.50 + 0.50 * road_area) if scene_areas else asphalt_clip
    )

    # 8. Екшн, гонки, вибухи
    combined["combat_action"] = source_scores.get("combat_action", 0.0)
    combined["racing_speed"] = source_scores.get("racing_speed", 0.0)
    combined["explosions"] = source_scores.get("explosions", 0.0)

    return {source_name: clamp(value) for source_name, value in combined.items()}


def apply_source_interactions(
    sources: ScoreMapping,
    scene_areas: ScoreMapping,
) -> dict[str, float]:
    """Застосовує взаємодії та крос-перевірку між YOLO та CLIP."""
    result = {name: clamp(value) for name, value in sources.items()}

    # 1. Пригнічення бруду, якщо немає фізичної землі
    soil = result.get("soil_mud", 0.0)
    ground_area = scene_areas.get("ground", 0.0)
    if scene_areas and ground_area < 0.02:
        soil *= 0.50
    result["soil_mud"] = clamp(soil)

    # 2. Фільтрація хибного диму/вибухів (якщо YOLO не виявила вогню/диму)
    fire_hazard = scene_areas.get("fire_hazard", 0.0)
    if scene_areas and fire_hazard < 0.01:
        # Пригнічуємо помилкові спрацювання CLIP для диму та вибухів
        result["smoke_fire"] = clamp(result.get("smoke_fire", 0.0) * 0.15)
        result["explosions"] = 0.0

    return result


def sources_to_cartridges(
    source_scores: ScoreMapping,
) -> dict[str, float]:
    """Перетворює джерела запаху на фізичні картриджі."""
    cartridges: defaultdict[str, float] = defaultdict(float)

    for source_name, source_score in source_scores.items():
        clean_key = str(source_name).lower().strip().replace(" ", "_")
        mapping = SOURCE_TO_CARTRIDGE.get(clean_key, {})

        for cartridge_name, weight in mapping.items():
            cartridges[cartridge_name] += float(source_score) * float(weight)

    return {cartridge_name: clamp(value) for cartridge_name, value in cartridges.items()}


def extract_object_areas(
    segments: list[dict[str, Any]],
    min_confidence: float = 0.50,
) -> dict[str, float]:
    """Перетворює детекції/маски в площі об'єктів."""
    object_scores: defaultdict[str, float] = defaultdict(float)

    for segment in segments:
        raw_label = segment.get("label")
        if raw_label is None:
            continue

        label = str(raw_label).strip().lower()
        if not label:
            continue

        confidence = safe_float(segment.get("score", segment.get("confidence", 1.0)))
        if confidence < min_confidence:
            continue

        mask = segment.get("mask")
        if mask is not None:
            arr = np.asarray(mask)
            if arr.size == 0:
                continue
            area_ratio = safe_float((arr > 0).mean())
        else:
            area_ratio = safe_float(segment.get("area_ratio", 0.0))

        object_scores[label] += area_ratio * confidence

    return {
        label: clamp(score)
        for label, score in object_scores.items()
    }


def apply_cartridge_thresholds(
    cartridges: ScoreMapping,
    threshold: float = 0.10,
) -> dict[str, float]:
    """Вимикає картриджі з інтенсивністю нижче порога."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    return {
        name: (clamp(value) if value >= threshold else 0.0)
        for name, value in cartridges.items()
    }


def analyze_scent_image(
    image: Image.Image,
    segmenter: Any,
    classifier: Any,
    threshold: float = 0.10,
    min_confidence: float = 0.50,
) -> dict[str, Any]:
    """Запускає повний пайплайн аналізу кадрів."""
    if not isinstance(image, Image.Image):
        raise TypeError("analyze_scent_image() expects a PIL.Image.Image.")

    image = image.convert("RGB")

    # 1. Отримуємо детекції
    segments = segmenter(image) if segmenter else []
    object_areas = extract_object_areas(
        segments=list(segments),
        min_confidence=min_confidence,
    )

    # 2. Семантичні групи
    scene_areas = build_scene_areas(object_areas)

    # 3. Оцінка CLIP
    source_scores: dict[str, float] = {}
    source_details: dict[str, SourceResult] = {}

    for source_name, source_config in SCENT_SOURCES.items():
        result = classify_source(
            image=image,
            classifier=classifier,
            config=source_config,
        )
        source_scores[source_name] = float(result["score"])
        source_details[source_name] = result

    # 4. Об'єднання CLIP + YOLO
    combined_sources = combine_sources_with_segmentation(
        source_scores=source_scores,
        scene_areas=scene_areas,
    )

    # 5. Крос-валідація та взаємодії
    final_sources = apply_source_interactions(
        sources=combined_sources,
        scene_areas=scene_areas,
    )

    # 6. Мапінг у картриджі
    mapped_cartridges = sources_to_cartridges(final_sources)

    cartridges = {
        cartridge: mapped_cartridges.get(cartridge, 0.0)
        for cartridge in CARTRIDGES
    }

    # 7. Порогова фільтрація
    cartridges = apply_cartridge_thresholds(
        cartridges,
        threshold=threshold,
    )

    return {
        "object_areas": object_areas,
        "scene_areas": scene_areas,
        "clip_sources": source_scores,
        "source_details": source_details,
        "combined_sources": combined_sources,
        "final_sources": final_sources,
        "cartridges": cartridges,
    }