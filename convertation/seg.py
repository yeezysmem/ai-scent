# export_segformer_onnx.py
from __future__ import annotations

import torch
from pathlib import Path
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


def export_segformer_to_onnx(
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    output_dir: str = "./models/segformer_onnx",
    image_size: int = 512,
    opset_version: int = 14,
):
    """
    Експортує SegFormer в ONNX формат.
    
    Модель приймає: pixel_values (batch, 3, H, W)
    Модель повертає: logits (batch, num_classes, H/4, W/4)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"→ Завантаження моделі: {model_name}")
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    
    processor = SegformerImageProcessor.from_pretrained(model_name)
    
    # Створюємо dummy input для експорту
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    onnx_path = output_dir / "segformer.onnx"
    
    print(f"→ Експорт в ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    # Зберігаємо processor (потрібен для препроцесингу)
    processor.save_pretrained(str(output_dir))
    
    # Зберігаємо config моделі (для id2label)
    model.config.save_pretrained(str(output_dir))
    
    file_size = onnx_path.stat().st_size / 1024 / 1024
    print(f"✓ Збережено: {onnx_path} ({file_size:.1f} MB)")
    
    return onnx_path


if __name__ == "__main__":
    export_segformer_to_onnx()