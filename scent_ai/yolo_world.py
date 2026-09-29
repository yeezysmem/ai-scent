from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class YOLOWorldDetector:
    """YOLO-World детектор на базі ONNX Runtime зі збереженим API."""

    def __init__(
        self,
        model_path: str = "yolov8s-world.onnx",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        img_size: int = 672,
    ) -> None:
        """
        Ініціалізація ONNX YOLO-World детектора.
        
        Args:
            model_path: Шлях до експортованого ONNX файлу
            confidence_threshold: Поріг впевненості для виявлення
            iou_threshold: Поріг IoU для NMS
            img_size: Розмір вхідного тензора моделі (за замовчуванням 672)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Порядок класів ПОБИНЕН точно відповідати масиву зі скрипту експорту
        self.default_classes = [
            "tree", "pine tree", "conifer tree", "plant", "forest",
            "sea", "ocean", "lake", "river", "water", "wave",
            "fire", "smoke", "flame", "bonfire", "exhaust smoke",
            "dirt", "soil", "mud", "sand", "grass", "rock", "ground",
            "cloud", "fog", "puddle", "rain",
            "road", "asphalt", "street", "car", "truck", "pavement",
        ]

        # Автоматичний вибір прискорювача
        providers = [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"✅ YOLO-World ONNX завантажено з {len(self.default_classes)} класами")

    def set_classes(self, classes: List[str]) -> None:
        """Метод залишено для сумісності з API (для ONNX класи запечені під час експорту)."""
        print("⚠️ Увага: класи в ONNX моделі запечені при експорті. Зміна через set_classes() не застосовується.")

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, float, float]:
        """Підготовка PIL зображення до формату [1, 3, 672, 672]."""
        orig_w, orig_h = image.size
        img_resized = image.convert("RGB").resize((self.img_size, self.img_size))

        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, float(orig_w), float(orig_h)

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Виявляє об'єкти на зображенні через ONNX.
        
        Returns:
            Dict з ключами:
            - objects: {class_name: [bbox1, bbox2, ...]}
            - areas: {class_name: total_area}
            - percentages: {class_name: percentage}
        """
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        tensor, orig_w, orig_h = self._preprocess(image_pil)

        # Інференс моделі
        outputs = self.session.run(None, {self.input_name: tensor})
        preds = np.squeeze(outputs[0]).T  # Перетворення у [8400, 4 + num_classes]

        boxes_cxcywh = preds[:, :4]
        scores_matrix = preds[:, 4:]

        class_ids = np.argmax(scores_matrix, axis=1)
        confidences = np.max(scores_matrix, axis=1)

        # Первинний фільтр за порогом впевненості
        mask = confidences >= self.confidence_threshold
        filtered_boxes_cxcywh = boxes_cxcywh[mask]
        filtered_confs = confidences[mask]
        filtered_classes = class_ids[mask]

        img_area = int(orig_w * orig_h)

        if len(filtered_confs) == 0:
            return {
                "objects": {},
                "areas": {},
                "percentages": {},
                "total_area": 0,
                "image_area": img_area,
                "num_objects": 0,
                "num_classes": 0,
            }

        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size

        nms_boxes = []
        for box in filtered_boxes_cxcywh:
            cx, cy, w, h = box
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            bw = w * scale_x
            bh = h * scale_y
            nms_boxes.append([int(x1), int(y1), int(bw), int(bh)])

        # Застосування Non-Maximum Suppression через OpenCV
        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=filtered_confs.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.iou_threshold,
        )

        objects: Dict[str, List[Dict[str, Any]]] = {}
        areas: Dict[str, float] = {}
        total_area = 0.0

        if len(indices) > 0:
            for idx in indices.flatten():
                cls_id = int(filtered_classes[idx])
                if cls_id >= len(self.default_classes):
                    continue

                cls_name = self.default_classes[cls_id]
                conf = float(filtered_confs[idx])

                x1, y1, bw, bh = nms_boxes[idx]
                x2 = int(x1 + bw)
                y2 = int(y1 + bh)

                # Кліппінг у межах кадру
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(int(orig_w), x2), min(int(orig_h), y2)

                width = x2 - x1
                height = y2 - y1
                area = float(width * height)

                if cls_name not in objects:
                    objects[cls_name] = []

                objects[cls_name].append({
                    "bbox": [x1, y1, x2, y2],
                    "width": width,
                    "height": height,
                    "area": area,
                    "confidence": conf,
                })

                areas[cls_name] = areas.get(cls_name, 0.0) + area
                total_area += area

        percentages: Dict[str, float] = {}
        if total_area > 0:
            for cls_name, area in areas.items():
                percentages[cls_name] = area / total_area

        return {
            "objects": objects,
            "areas": areas,
            "percentages": percentages,
            "total_area": total_area,
            "image_area": img_area,
            "num_objects": sum(len(v) for v in objects.values()),
            "num_classes": len(objects),
        }

    def get_object_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Вирізає кожен знайдений об'єкт для окремого аналізу CLIP."""
        detection = self.detect(image)
        regions = []

        for class_name, boxes in detection["objects"].items():
            for box in boxes:
                bbox = box["bbox"]
                cropped = image.crop(bbox)

                regions.append({
                    "class_name": class_name,
                    "cropped": cropped,
                    "bbox": bbox,
                    "confidence": box["confidence"],
                    "area": box["area"],
                })

        return regions

    def get_top_classes(self, image: Image.Image, top_n: int = 10) -> List[Tuple[str, float]]:
        """Повертає топ-N класів за площею."""
        results = self.detect(image)
        percentages = results["percentages"]
        sorted_classes = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[:top_n]

    def get_cartridge_weights(self, image: Image.Image) -> Dict[str, float]:
        """Перетворює виявлені об'єкти у сирі ваги для 6 картриджів."""
        results = self.detect(image)
        percentages = results["percentages"]

        cartridge_weights = {
            "pine": 0.0,
            "ocean": 0.0,
            "smoke": 0.0,
            "earth": 0.0,
            "rain": 0.0,
            "asphalt": 0.0,
        }

        class_mapping = {
            # 🌲 Pine
            "tree": {"pine": 0.8, "earth": 0.2},
            "pine tree": {"pine": 1.0},
            "conifer tree": {"pine": 1.0},
            "plant": {"pine": 0.5, "earth": 0.3},
            "forest": {"pine": 0.9, "earth": 0.3},

            # 🌊 Ocean
            "sea": {"ocean": 1.0},
            "ocean": {"ocean": 1.0},
            "lake": {"ocean": 0.7, "rain": 0.2},
            "river": {"ocean": 0.6, "rain": 0.2},
            "water": {"ocean": 0.8},
            "wave": {"ocean": 0.9},

            # 🔥 Smoke
            "fire": {"smoke": 1.0},
            "smoke": {"smoke": 1.0},
            "flame": {"smoke": 1.0},
            "bonfire": {"smoke": 1.0},
            "exhaust smoke": {"smoke": 0.9, "asphalt": 0.3},

            # 🪵 Earth
            "dirt": {"earth": 0.9},
            "soil": {"earth": 1.0},
            "mud": {"earth": 0.9, "rain": 0.3},
            "sand": {"earth": 0.7},
            "grass": {"earth": 0.6, "pine": 0.2},
            "rock": {"earth": 0.5},
            "ground": {"earth": 0.7},

            # 🌧️ Rain
            "cloud": {"rain": 0.4, "ocean": 0.2},
            "fog": {"rain": 0.6},
            "puddle": {"rain": 0.9, "asphalt": 0.2},
            "rain": {"rain": 1.0},

            # 🛣️ Asphalt
            "road": {"asphalt": 1.0},
            "asphalt": {"asphalt": 1.0},
            "street": {"asphalt": 0.9},
            "pavement": {"asphalt": 0.8},
            "car": {"asphalt": 0.5, "smoke": 0.1},
            "truck": {"asphalt": 0.6, "smoke": 0.2},
        }

        for class_name, percentage in percentages.items():
            if class_name in class_mapping:
                for cartridge, weight in class_mapping[class_name].items():
                    if cartridge in cartridge_weights:
                        cartridge_weights[cartridge] += percentage * weight

        return cartridge_weights

    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        return self.detect(image)