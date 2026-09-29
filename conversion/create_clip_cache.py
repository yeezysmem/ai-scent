# conversion/create_clip_cache.py
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

def create_clip_cache():
    # Завантажуємо модель CLIP
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # ВАШІ КАТЕГОРІЇ З КОНФІГУ (10 джерел)
    labels = [
        "coniferous_forest",
        "broadleaf_forest",
        "ocean_sea",
        "rain_wetness",
        "smoke_fire",
        "soil_mud",
        "asphalt_city",
        "combat_action",
        "racing_speed",
        "explosions",
    ]
    
    # Генеруємо текстові ембединги
    print("🔄 Генерація текстових ембедингів...")
    texts = [f"a photo of {label}" for label in labels]
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    
    # Зберігаємо кеш
    cache = {
        "labels": labels,
        "text_embeds": text_embeds
    }
    
    output_path = Path("models/clip_onnx/cached_text_embeds.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    
    print(f"✅ Кеш збережено в {output_path}")
    print(f"📊 Збережено {len(labels)} категорій:")
    for i, label in enumerate(labels):
        print(f"  {i+1}. {label}")

if __name__ == "__main__":
    create_clip_cache()