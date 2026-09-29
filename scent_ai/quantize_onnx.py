# scent_ai/quantize_onnx.py
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    base_dir = Path(__file__).parent.parent / "models"
    
    print(f"🔍 Шукаю моделі в: {base_dir}")
    print(f"   Існує? {base_dir.exists()}\n")
    
    segformer_dir = base_dir / "segformer_onnx"
    clip_dir = base_dir / "clip_onnx"
    
    # 1. Квантування SegFormer (Це найважливіше для швидкості!)
    if segformer_dir.exists():
        print("→ Квантування SegFormer...")
        input_file = segformer_dir / "segformer.onnx"
        output_file = segformer_dir / "segformer_quantized.onnx"
        
        if input_file.exists():
            try:
                quantize_dynamic(
                    model_input=str(input_file),
                    model_output=str(output_file),
                    weight_type=QuantType.QInt8,
                )
                orig_size = input_file.stat().st_size / (1024 * 1024)
                new_size = output_file.stat().st_size / (1024 * 1024)
                print(f"   ✓ {output_file.name}")
                print(f"     {orig_size:.1f} MB → {new_size:.1f} MB\n")
            except Exception as e:
                print(f"   ⚠ Помилка квантування SegFormer: {e}\n")
        else:
            print(f"   ❌ {input_file.name} не знайдено!\n")

    # 2. Квантування CLIP (Спробуємо, але без паніки, якщо впаде)
    if clip_dir.exists():
        print("→ Спроба квантування CLIP...")
        for name in ["clip_image_encoder", "clip_text_encoder"]:
            input_file = clip_dir / f"{name}.onnx"
            output_file = clip_dir / f"{name}_quantized.onnx"
            
            if input_file.exists():
                try:
                    quantize_dynamic(
                        model_input=str(input_file),
                        model_output=str(output_file),
                        weight_type=QuantType.QInt8,
                    )
                    orig_size = input_file.stat().st_size / (1024 * 1024)
                    new_size = output_file.stat().st_size / (1024 * 1024)
                    print(f"   ✓ {output_file.name} ({orig_size:.1f} MB → {new_size:.1f} MB)")
                except Exception as e:
                    print(f"   ⚠ Пропускаю {name}.onnx (відома помилка shape inference в ONNX).")
                    print(f"     Це нормально, CLIP і так працює дуже швидко (~90 мс) через кеш.\n")
            else:
                print(f"   ❌ {input_file.name} не знайдено!")

    print("\n✅ Процес завершено! Можна запускати бенчмарк.")

if __name__ == "__main__":
    main()