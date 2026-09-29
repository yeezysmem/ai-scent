from scent_ai.capture import ScreenCapture


def main() -> None:
    with ScreenCapture(
        monitor_index=1,
    ) as capture:
        image = capture.capture()

    image.save("captured_screen.jpg")

    print("Screenshot saved:")
    print("captured_screen.jpg")
    print("Size:", image.size)


if __name__ == "__main__":
    main()