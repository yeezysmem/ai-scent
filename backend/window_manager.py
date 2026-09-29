from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from AppKit import NSWorkspace
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListExcludeDesktopElements,
    kCGWindowListOptionOnScreenOnly,
)


@dataclass(frozen=True)
class WindowInfo:
    id: int
    title: str
    owner: str
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MacOSWindowManager:
    """
    Отримує інформацію про відкриті вікна на macOS.
    """

    MIN_WINDOW_WIDTH = 300
    MIN_WINDOW_HEIGHT = 200

    @staticmethod
    def _read_bounds(window: dict[str, Any]) -> dict[str, int]:
        bounds = window.get("kCGWindowBounds", {})

        return {
            "x": int(bounds.get("X", 0)),
            "y": int(bounds.get("Y", 0)),
            "width": int(bounds.get("Width", 0)),
            "height": int(bounds.get("Height", 0)),
        }

    def list_windows(self) -> list[WindowInfo]:
        options = (
            kCGWindowListOptionOnScreenOnly
            | kCGWindowListExcludeDesktopElements
        )

        raw_windows = CGWindowListCopyWindowInfo(
            options,
            kCGNullWindowID,
        )

        if not raw_windows:
            return []

        windows: list[WindowInfo] = []

        for raw_window in raw_windows:
            window = dict(raw_window)

            window_id = int(
                window.get("kCGWindowNumber", 0),
            )

            owner = str(
                window.get("kCGWindowOwnerName", ""),
            ).strip()

            title = str(
                window.get("kCGWindowName", ""),
            ).strip()

            layer = int(
                window.get("kCGWindowLayer", 0),
            )

            alpha = float(
                window.get("kCGWindowAlpha", 1.0),
            )

            bounds = self._read_bounds(window)

            # Звичайні вікна програм зазвичай знаходяться на layer 0.
            if layer != 0:
                continue

            if alpha <= 0:
                continue

            if window_id <= 0:
                continue

            if not owner:
                continue

            if (
                bounds["width"] < self.MIN_WINDOW_WIDTH
                or bounds["height"] < self.MIN_WINDOW_HEIGHT
            ):
                continue

            display_title = title or owner

            windows.append(
                WindowInfo(
                    id=window_id,
                    title=display_title,
                    owner=owner,
                    x=bounds["x"],
                    y=bounds["y"],
                    width=bounds["width"],
                    height=bounds["height"],
                ),
            )

        return windows

    def get_window(
        self,
        window_id: int,
    ) -> WindowInfo | None:
        for window in self.list_windows():
            if window.id == window_id:
                return window

        return None

    def get_window_bounds(self, window_id: int) -> dict[str, int] | None:
        """
        Повертає координати вікна у форматі, необхідному для mss:
        {"left": X, "top": Y, "width": W, "height": H}
        """
        window = self.get_window(window_id)
        if window is None:
            return None

        return {
            "left": window.x,
            "top": window.y,
            "width": window.width,
            "height": window.height,
        }

    def get_window_geometry(self, window_id: int) -> dict[str, int] | None:
        """Аліас для сумісності з іншими модулями."""
        return self.get_window_bounds(window_id)

    def window_exists(self, window_id: int) -> bool:
        return self.get_window(window_id) is not None

    @staticmethod
    def get_frontmost_app_name() -> str | None:
        workspace = NSWorkspace.sharedWorkspace()
        application = workspace.frontmostApplication()

        if application is None:
            return None

        name = application.localizedName()

        if name is None:
            return None

        return str(name)

    def is_window_active(self, window_id: int) -> bool:
        window = self.get_window(window_id)

        if window is None:
            return False

        active_app = self.get_frontmost_app_name()

        if active_app is None:
            return False

        return active_app.casefold() == window.owner.casefold()