from pathlib import Path

from PIL import Image

from scent_ai.clip_classifier import ClipClassifier
from scent_ai.runtime import RuntimeSmellEngine
from scent_ai.segmentation import SegmentationModel
from scent_ai.smell_engine import print_scent_analysis


IMAGE_PATH = Path("test.jpg")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено зображення: {IMAGE_PATH.resolve()}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print("Loading SegFormer...")
    segmenter = SegmentationModel()

    print("Loading CLIP...")
    classifier = ClipClassifier()

    runtime = RuntimeSmellEngine(
        segmenter=segmenter,
        classifier=classifier,
        activation_threshold=0.10,
        smoothing_alpha=0.35,
        change_threshold=0.05,
        update_interval=1.0,
        max_active_cartridges=3,
    )

    print("Analyzing image...")

    result = runtime.process(
        image,
        force=True,
    )

    analysis = result["analysis"]

    print_scent_analysis(
        analysis
    )

    print("\n--- Runtime після згладжування ---")

    for cartridge, value in result["cartridges"].items():
        print(f"{cartridge:20s}: {value:.3f}")

    print("\n--- Значення у відсотках ---")
    print(runtime.get_percent_state())

    print("\n--- PWM для ESP32 ---")
    print(runtime.get_pwm_state())


if __name__ == "__main__":
    main()