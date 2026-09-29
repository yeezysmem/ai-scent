# benchmark_onnx.py
from __future__ import annotations

import time
from PIL import Image

from scent_ai.segmentation import SegmentationModel as SegPT
from scent_ai.segmentation_onnx import SegmentationModel as SegONNX
from scent_ai.clip_classifier import ClipClassifier as ClipPT
from scent_ai.clip_classifier_onnx import ClipClassifier as ClipONNX


def benchmark():
    from PIL import Image
    import time
    from pathlib import Path
    
    image = Image.open("scent_ai/test.jpeg").convert("RGB")
    
    labels = [
        "coniferous trees, pine forest, spruce",
        "indoor, room, interior",
    ]
    
    print("=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    
    # --- SegFormer ---
    print("\n[SegFormer]")
    
    from scent_ai.segmentation import SegmentationModel as SegPT
    model_pt = SegPT()
    model_pt(image)
    t0 = time.time()
    for _ in range(5): model_pt(image)
    print(f"  PyTorch:  {(time.time() - t0)/5*1000:.1f} мс")
    
    from scent_ai.segmentation_onnx import SegmentationModel as SegONNX
    model_fp32 = SegONNX(quantized=False)
    model_fp32(image)
    t0 = time.time()
    for _ in range(5): model_fp32(image)
    print(f"  ONNX FP32: {(time.time() - t0)/5*1000:.1f} мс")
    
    model_int8 = SegONNX(quantized=True)
    model_int8(image)
    t0 = time.time()
    for _ in range(5): model_int8(image)
    print(f"  ONNX INT8: {(time.time() - t0)/5*1000:.1f} мс (Рекомендовано)")

    # --- CLIP ---
    print("\n[CLIP]")
    
    from scent_ai.clip_classifier import ClipClassifier as ClipPT
    model_pt = ClipPT()
    model_pt(image, candidate_labels=labels)
    t0 = time.time()
    for _ in range(10): model_pt(image, candidate_labels=labels)
    print(f"  PyTorch:  {(time.time() - t0)/10*1000:.1f} мс")
    
    from scent_ai.clip_classifier_onnx import ClipClassifier as ClipONNX
    model_fp32 = ClipONNX(quantized=False)
    model_fp32(image, candidate_labels=labels)
    t0 = time.time()
    for _ in range(10): model_fp32(image, candidate_labels=labels)
    print(f"  ONNX FP32: {(time.time() - t0)/10*1000:.1f} мс (Рекомендовано)")
    
    # Безпечна перевірка для CLIP INT8
    clip_int8_path = Path(__file__).parent / "models" / "clip_onnx" / "clip_image_encoder_quantized.onnx"
    if clip_int8_path.exists():
        model_int8 = ClipONNX(quantized=True)
        model_int8(image, candidate_labels=labels)
        t0 = time.time()
        for _ in range(10): model_int8(image, candidate_labels=labels)
        print(f"  ONNX INT8: {(time.time() - t0)/10*1000:.1f} мс")
    else:
        print("  ONNX INT8: Пропущено (файл не створено через особливість ONNX)")
        print("  (Це нормально, CLIP FP32 і так працює дуже швидко)")

    print("\n" + "=" * 60)
    print("✅ Бенчмарк успішно завершено!")

if __name__ == "__main__":
    benchmark()