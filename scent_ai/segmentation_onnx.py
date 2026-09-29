from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor


class SegmentationModel:
    def __init__(
        self,
        model_dir: str,
        quantized: bool = False,
    ) -> None:
        model_dir = Path(model_dir)
        suffix = "_quantized" if quantized else ""
        
        onnx_file = model_dir / f"segformer{suffix}.onnx"
        
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX модель не знайдена: {onnx_file}")
        
        self._session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self._processor = SegformerImageProcessor.from_pretrained(str(model_dir))
        
        # ===== ВСІ 150 КЛАСІВ ADE20K =====
        self.id2label = {
            0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "road",
            5: "sidewalk", 6: "earth", 7: "grass", 8: "tree", 9: "plant",
            10: "water", 11: "sea", 12: "river", 13: "sand", 14: "mountain",
            15: "car", 16: "truck", 17: "bus", 18: "train", 19: "motorcycle",
            20: "bicycle", 21: "person", 22: "fence", 23: "railing", 24: "sign",
            25: "traffic_light", 26: "traffic_sign", 27: "pole", 28: "streetlight",
            29: "bench", 30: "chair", 31: "table", 32: "screen", 33: "door",
            34: "window", 35: "bridge", 36: "tunnel", 37: "staircase",
            38: "escalator", 39: "elevator", 40: "shelf", 41: "cabinet",
            42: "counter", 43: "sofa", 44: "bed", 45: "wardrobe", 46: "lamp",
            47: "painting", 48: "mirror", 49: "curtain", 50: "pillow",
            51: "blanket", 52: "towel", 53: "plate", 54: "bowl", 55: "cup",
            56: "bottle", 57: "can", 58: "box", 59: "bag", 60: "ball",
            61: "toy", 62: "book", 63: "paper", 64: "pen", 65: "phone",
            66: "computer", 67: "keyboard", 68: "mouse", 69: "camera",
            70: "laptop", 71: "speaker", 72: "microphone", 73: "headphone",
            74: "watch", 75: "clock", 76: "umbrella", 77: "hat", 78: "glasses",
            79: "shoes", 80: "clothes", 81: "fire", 82: "smoke", 83: "fog",
            84: "rain", 85: "snow", 86: "ice", 87: "rock", 88: "stone",
            89: "wood", 90: "metal", 91: "glass", 92: "plastic", 93: "paper",
            94: "food", 95: "fruit", 96: "vegetable", 97: "flower", 98: "leaf",
            99: "branch", 100: "stem", 101: "root", 102: "seed", 103: "fungus",
            104: "animal", 105: "bird", 106: "fish", 107: "insect", 108: "spider",
            109: "reptile", 110: "amphibian", 111: "mammal", 112: "human",
            113: "child", 114: "adult", 115: "elder", 116: "crowd",
            117: "group", 118: "pair", 119: "single", 120: "object",
            121: "container", 122: "furniture", 123: "vehicle", 124: "building",
            125: "room", 126: "hall", 127: "corridor", 128: "entrance",
            129: "exit", 130: "window", 131: "door", 132: "roof",
            133: "floor", 134: "ceiling", 135: "wall", 136: "column",
            137: "step", 138: "stair", 139: "ramp", 140: "bridge",
            141: "tunnel", 142: "platform", 143: "track", 144: "rail",
            145: "road", 146: "sidewalk", 147: "path", 148: "trail",
            149: "street"
        }
        
        # ===== АГРЕГАЦІЯ В 7 КАТЕГОРІЙ =====
        self.category_mapping = {
            "road": [4, 145, 146, 147, 148, 149],
            "sidewalk": [5, 146],
            "building": [1, 124, 125, 126, 127, 128, 129, 132, 135],
            "vegetation": [7, 8, 9, 97, 98, 99, 100, 101, 102, 103],
            "sky": [2],
            "car": [15, 16, 17, 18, 123],
            "terrain": [6, 13, 14, 87, 88],
        }
        
        self.categories = list(self.category_mapping.keys())
        
        # Генеруємо кольори для 150 класів
        self.colors = self._generate_colors(150)
    
    def _generate_colors(self, num_classes: int) -> list[tuple[int, int, int]]:
        """Генерує кольори для кожного класу"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def predict(self, image: Image.Image) -> np.ndarray:
        """Повертає маску сегментації як np.ndarray"""
        inputs = self._processor(images=image, return_tensors="np")
        outputs = self._session.run(None, {"pixel_values": inputs["pixel_values"]})
        logits = outputs[0]
        
        logits_tensor = torch.from_numpy(logits)
        upsampled_logits = F.interpolate(
            logits_tensor,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        
        mask = upsampled_logits.argmax(dim=1).squeeze(0).numpy()
        return mask

    def __call__(self, image: Image.Image) -> list[dict[str, Any]]:
        """Повертає список сегментів у форматі, очікуваному smell_engine"""
        mask = self.predict(image)
        return self._postprocess(mask, image.size)

    def _postprocess(self, mask: np.ndarray, original_size: tuple[int, int]) -> list[dict[str, Any]]:
        """Перетворює маску в список сегментів, агрегуючи 150 класів у 7 категорій"""
        segments = []
        total_pixels = mask.size
        
        # Агрегуємо класи в категорії
        for category, class_ids in self.category_mapping.items():
            # Об'єднуємо всі класи для цієї категорії
            combined_mask = np.zeros_like(mask, dtype=np.uint8)
            for class_id in class_ids:
                combined_mask = np.logical_or(combined_mask, (mask == class_id))
            combined_mask = combined_mask.astype(np.uint8) * 255
            
            area = np.sum(combined_mask) / 255
            if area > 0:
                percentage = (area / total_pixels) * 100
                # Беремо колір першого класу в категорії
                first_class_id = self.category_mapping[category][0]
                color = self.colors[first_class_id] if first_class_id < len(self.colors) else (128, 128, 128)
                
                segments.append({
                    "label": category,
                    "mask": Image.fromarray(combined_mask),
                    "color": color,
                    "area": int(area),
                    "percentage": percentage
                })
        
        segments.sort(key=lambda x: x["area"], reverse=True)
        return segments

    def get_class_stats(self, mask: np.ndarray) -> dict[str, float]:
        """Повертає відсоток кожної категорії на зображенні"""
        total = mask.size
        stats = {}
        
        for category, class_ids in self.category_mapping.items():
            combined_mask = np.zeros_like(mask, dtype=bool)
            for class_id in class_ids:
                combined_mask = np.logical_or(combined_mask, (mask == class_id))
            count = np.sum(combined_mask)
            stats[category] = count / total
        
        return stats

    def visualize(self, mask: np.ndarray, original_image: Image.Image | None = None) -> Image.Image:
        """Створює кольорову візуалізацію маски з агрегованими категоріями"""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Спочатку малюємо всі класи
        for class_id, color in enumerate(self.colors):
            colored_mask[mask == class_id] = color
        
        if original_image is not None:
            original = np.array(original_image.resize((w, h)))
            blended = (original * 0.4 + colored_mask * 0.6).astype(np.uint8)
            return Image.fromarray(blended)
        
        return Image.fromarray(colored_mask)