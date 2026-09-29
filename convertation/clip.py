# clip.py
from __future__ import annotations

import torch
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor


def export_clip_to_onnx(
    model_name: str = "openai/clip-vit-base-patch32",
    output_dir: str = "./models/clip_onnx",
    image_size: int = 224,
    max_text_length: int = 77,
    opset_version: int = 14,
):
    """
    Експортує CLIP в ONNX формат (дві окремі моделі).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"→ Завантаження моделі: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # ---------------------------------------------------------
    # 1. Text Encoder
    # ---------------------------------------------------------
    print("\n→ Експорт Text Encoder...")
    
    text_encoder = model.text_model
    text_encoder.eval()
    
    # Dummy inputs для text encoder
    dummy_input_ids = torch.zeros((1, max_text_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_text_length), dtype=torch.long)
    
    text_onnx_path = output_dir / "clip_text_encoder.onnx"
    
    torch.onnx.export(
        text_encoder,
        (dummy_input_ids, dummy_attention_mask),
        str(text_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embeds": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"✓ Text Encoder: {text_onnx_path}")
    
    # ---------------------------------------------------------
    # 2. Image Encoder
    # ---------------------------------------------------------
    print("\n→ Експорт Image Encoder...")
    
    # Обгортаємо vision_model + projection в одну модель
    class ClipImageEncoderWrapper(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.vision_model = clip_model.vision_model
            self.visual_projection = clip_model.visual_projection
        
        def forward(self, pixel_values):
            vision_output = self.vision_model(pixel_values=pixel_values)
            pooled = vision_output[1]  # pooled_output
            image_embeds = self.visual_projection(pooled)
            return image_embeds
    
    image_encoder = ClipImageEncoderWrapper(model)
    image_encoder.eval()
    
    dummy_image = torch.randn(1, 3, image_size, image_size)
    
    image_onnx_path = output_dir / "clip_image_encoder.onnx"
    
    torch.onnx.export(
        image_encoder,
        dummy_image,
        str(image_onnx_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"✓ Image Encoder: {image_onnx_path}")
    
    # ---------------------------------------------------------
    # 3. Зберігаємо processor і config
    # ---------------------------------------------------------
    print("\n→ Збереження processor та config...")
    processor.save_pretrained(str(output_dir))
    model.config.save_pretrained(str(output_dir))
    
    # ---------------------------------------------------------
    # 4. Кешуємо text embeddings для стандартних label-ів
    # ---------------------------------------------------------
    print("\n→ Попереднє кодування стандартних label-ів...")
    
    # Label-и з твого smell engine
    default_labels = [
        "coniferous trees, pine forest, spruce",
        "forest vegetation, dense woods, green plants",
        "ocean water, sea waves, blue water",
        "smoke and fire, flames, burning",
        "soil, dirt, earth, ground",
        "rain, wetness, water drops",
        "asphalt, road surface, pavement",
        "flowers, blooming plants, petals",
    ]
    
    negative_labels = [
        "indoor, room, interior",
        "abstract, pattern, texture",
        "empty, nothing, blank",
    ]
    
    # Кешуємо text embeddings (щоб не кодувати їх щоразу)
    with torch.no_grad():
        all_labels = default_labels + negative_labels
        inputs = processor(
            text=all_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_embeds = model.get_text_features(**inputs)
    
    torch.save(
        {
            "labels": all_labels,
            "positive_labels": default_labels,
            "negative_labels": negative_labels,
            "text_embeds": text_embeds,
        },
        output_dir / "cached_text_embeds.pt",
    )
    
    print(f"✓ Кеш text embeddings: {output_dir / 'cached_text_embeds.pt'}")
    
    # ---------------------------------------------------------
    # 5. Підсумок
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("✓ CLIP успішно експортовано!")
    print("=" * 60)
    print(f"\nФайли:")
    print(f"  - {text_onnx_path}")
    print(f"  - {image_onnx_path}")
    print(f"  - {output_dir / 'cached_text_embeds.pt'}")
    print(f"  - {output_dir / 'preprocessor_config.json'}")
    print(f"  - {output_dir / 'config.json'}")
    
    return text_onnx_path, image_onnx_path


if __name__ == "__main__":
    export_clip_to_onnx()
    