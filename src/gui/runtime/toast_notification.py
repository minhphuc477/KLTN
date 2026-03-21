"""Floating toast notification model and renderer for GUI overlays."""

from typing import Any


class ToastNotification:
    """Floating toast message with auto-dismiss and fade animations."""

    def __init__(self, message: str, duration: float = 3.0, toast_type: str = "info", time_module: Any = None):
        self.message = message
        self.duration = duration
        self.toast_type = toast_type  # info | success | error | warning
        self._time = time_module if time_module is not None else __import__("time")
        self.created_at = self._time.time()

        self.colors = {
            "info": (100, 200, 255),
            "success": (100, 255, 150),
            "error": (255, 100, 100),
            "warning": (255, 200, 100),
        }

    def is_expired(self) -> bool:
        """Check if toast should be removed."""
        return self._time.time() - self.created_at > self.duration

    def get_alpha(self) -> int:
        """Calculate current alpha for fade in/out animation."""
        elapsed = self._time.time() - self.created_at
        alpha = 240

        if elapsed < 0.3:
            alpha = int((elapsed / 0.3) * 240)
        elif elapsed > self.duration - 0.5:
            remaining = self.duration - elapsed
            alpha = int((remaining / 0.5) * 240)

        return max(0, min(255, int(alpha)))

    def render(self, surface: Any, center_x: int, y: int, pygame_module: Any = None) -> None:
        """Render toast notification at specified position."""
        pygame = pygame_module if pygame_module is not None else __import__("pygame")
        alpha = self.get_alpha()
        font = pygame.font.SysFont("Arial", 20)
        text_surf = font.render(self.message, True, (255, 255, 255))

        padding = 15
        toast_w = text_surf.get_width() + padding * 2
        toast_h = text_surf.get_height() + padding * 2

        toast_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
        bg_rect = pygame.Rect(0, 0, toast_w, toast_h)

        try:
            pygame.draw.rect(toast_surf, (50, 60, 80, int(alpha)), bg_rect, border_radius=8)
        except Exception:
            pygame.draw.rect(toast_surf, (50, 60, 80), bg_rect, border_radius=8)

        col = self.colors.get(self.toast_type, (200, 200, 200))
        try:
            pygame.draw.rect(
                toast_surf,
                (int(col[0]), int(col[1]), int(col[2]), int(alpha)),
                bg_rect,
                2,
                border_radius=8,
            )
        except Exception:
            pygame.draw.rect(toast_surf, (200, 200, 200), bg_rect, 2, border_radius=8)

        text_with_alpha = text_surf.copy()
        try:
            text_with_alpha.set_alpha(int(alpha))
        except Exception:
            pass
        toast_surf.blit(text_with_alpha, (padding, padding))

        surface.blit(toast_surf, (center_x - toast_w // 2, y))
