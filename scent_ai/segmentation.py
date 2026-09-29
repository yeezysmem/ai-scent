from __future__ import annotations

from typing import Any

from PIL import Image
from transformers import pipeline


DEFAULT_SEGMENTATION_MODEL = (
    "nvidia/segformer-b0-finetuned-ade-512-512"
)


class SegmentationModel:
    def __init__(
        self,
        model_name: str = DEFAULT_SEGMENTATION_MODEL,
        device: int = -1,
    ) -> None:
        """
        device:
            -1 = CPU
             0 = перша CUDA GPU
        """

        self.model_name = model_name
        self.device = device

        self._pipeline = pipeline(
            task="image-segmentation",
            model=model_name,
            device=device,
        )

    def predict(
        self,
        image: Image.Image,
    ) -> list[dict[str, Any]]:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "SegmentationModel.predict() expects a PIL image."
            )

        image = image.convert("RGB")

        results = self._pipeline(image)

        return list(results)

    def __call__(
        self,
        image: Image.Image,
    ) -> list[dict[str, Any]]:
        return self.predict(image)