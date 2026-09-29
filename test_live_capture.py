from scent_ai.capture import ScreenCapture
from scent_ai.clip_classifier import ClipClassifier
from scent_ai.runtime import RuntimeSmellEngine
from scent_ai.segmentation import SegmentationModel


def main() -> None:
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

    print("Capturing screen...")

    with ScreenCapture(
        monitor_index=1,
    ) as capture:
        image = capture.capture()

    print("Analyzing screenshot...")

    result = runtime.process(
        image,
        force=True,
    )

    print("\n--- Screenshot size ---")
    print(image.size)

    print("\n--- Runtime cartridges ---")

    for cartridge, value in result["cartridges"].items():
        print(f"{cartridge:20s}: {value:.3f}")

    print("\n--- Percent state ---")
    print(runtime.get_percent_state())

    print("\n--- PWM state ---")
    print(runtime.get_pwm_state())


if __name__ == "__main__":
    main()