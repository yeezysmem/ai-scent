from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import CLIPProcessor


def to_scalar(value: Any) -> float:
    """Гарантовано перетворює будь-яке значення у Python float."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (ValueError, TypeError):
            arr = np.asarray(value)
            if arr.size == 1:
                return float(arr.item())
            return float(arr.mean())
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr.item())
    return float(arr.mean())


class ClipClassifier:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        quantized: bool = False,
    ) -> None:
        if model_dir is None:
            # Автоматичний пошук папки з моделлю
            base_path = Path(__file__).parent.parent / "models" / "clip_onnx"
            model_dir = base_path if base_path.exists() else Path("models/clip_onnx")

        model_dir = Path(model_dir)
        suffix = "_quantized" if quantized else ""

        text_file = model_dir / f"clip_text_encoder{suffix}.onnx"
        image_file = model_dir / f"clip_image_encoder{suffix}.onnx"

        if not text_file.exists() or not image_file.exists():
            raise FileNotFoundError(f"ONNX CLIP моделі не знайдено в {model_dir}")

        self._text_session = ort.InferenceSession(str(text_file), providers=["CPUExecutionProvider"])
        self._image_session = ort.InferenceSession(str(image_file), providers=["CPUExecutionProvider"])
        self._processor = CLIPProcessor.from_pretrained(str(model_dir))

        cache_path = model_dir / "cached_text_embeds.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Кеш text embeddings не знайдено: {cache_path}")

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self._all_labels: list[str] = cache["labels"]

        raw_embeds = cache["text_embeds"]
        embeds_tensor = raw_embeds.pooler_output if hasattr(raw_embeds, "pooler_output") else raw_embeds
        self._all_text_embeds: np.ndarray = embeds_tensor.numpy()

        norms = np.linalg.norm(self._all_text_embeds, axis=1, keepdims=True)
        self._all_text_embeds_norm = self._all_text_embeds / np.maximum(norms, 1e-8)
        self._dynamic_cache: dict[str, np.ndarray] = {}

    def _get_text_embeds(self, labels: list[str]) -> np.ndarray:
        try:
            indices = [self._all_labels.index(lbl) for lbl in labels]
            embeds = self._all_text_embeds_norm[indices]
            
            # Страховка розмірності для кешованих ембеддингів
            if embeds.ndim == 3:
                embeds = embeds[:, 0, :]
            elif embeds.ndim == 2 and embeds.shape[1] != 512:
                embeds = embeds.reshape(-1, 77, 512)[:, 0, :]
                
            return embeds
        except ValueError:
            pass

        result = []
        for lbl in labels:
            if lbl in self._dynamic_cache:
                result.append(self._dynamic_cache[lbl])
            else:
                inputs = self._processor(
                    text=[lbl], 
                    return_tensors="np", 
                    padding="max_length", 
                    max_length=77, 
                    truncation=True
                )

                input_ids = inputs["input_ids"].astype(np.int64)
                attention_mask = inputs["attention_mask"].astype(np.int64)

                outputs = self._text_session.run(None, {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

                embed = np.array(outputs[0])

                # 🛠️ ГОЛОВНИЙ ФІКС: Якщо ONNX повертає [1, 77, 512] або [77, 512]
                if embed.ndim == 3:  # [batch, seq_len, hidden_dim] -> [1, 77, 512]
                    embed = embed[:, 0, :]  # Беремо [CLS] токен або 0-й індекс -> [1, 512]
                elif embed.ndim == 2 and embed.shape[0] == 77:  # [77, 512]
                    embed = embed[0, :]  # Беремо 0-й токен -> [512]
                elif embed.size == 39424:  # Якщо тензор розплющило у 1D 39424
                    embed = embed.reshape(77, 512)[0, :]  # Решейпимо та беремо 0-й токен -> [512]

                embed = np.squeeze(embed)  # Гарантуємо 1D вектор з 512 елементів

                # Нормалізація вектору
                norm = np.linalg.norm(embed)
                if norm > 1e-8:
                    embed = embed / norm

                self._dynamic_cache[lbl] = embed
                result.append(embed)

        return np.stack(result)

    def classify(self, image: Image.Image, candidate_labels: Sequence[str]) -> list[dict[str, Any]]:
        if not isinstance(image, Image.Image):
            raise TypeError("ClipClassifier.classify() expects a PIL image.")

        labels = list(candidate_labels)
        if not labels:
            raise ValueError("candidate_labels cannot be empty.")

        image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="np")

        if hasattr(inputs, "pixel_values"):
            pixel_values = inputs.pixel_values
        else:
            pixel_values = inputs["pixel_values"] if "pixel_values" in inputs else inputs.get("pixel_values")

        if hasattr(pixel_values, "numpy"):
            pixel_values = pixel_values.numpy()

        image_outputs = self._image_session.run(None, {"pixel_values": pixel_values})

        image_embed = np.squeeze(image_outputs[0])
        norm = np.linalg.norm(image_embed)
        if norm > 1e-8:
            image_embed = image_embed / norm

        text_embeds = self._get_text_embeds(labels)

        if image_embed.ndim == 1:
            similarity = text_embeds @ image_embed
        else:
            similarity = (text_embeds @ image_embed.T).flatten()

        logits = similarity * 100.0
        exp_logits = np.exp(logits - np.max(logits))
        probs = np.squeeze(exp_logits / np.sum(exp_logits))

        if np.isscalar(probs):
            probs = [probs]

        # Повертаємо класичний список словників [{'label': ..., 'score': ...}], 
        # який очікує більшість методів у smell_engine / runtime
        results = []
        for lbl, score in zip(labels, probs):
            results.append({"label": str(lbl), "score": to_scalar(score)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def __call__(self, image: Image.Image, *, candidate_labels: Sequence[str]) -> list[dict[str, Any]]:
        return self.classify(image=image, candidate_labels=candidate_labels)