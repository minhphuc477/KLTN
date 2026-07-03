"""Small pygame font cache for frame-rendered GUI overlays."""

from __future__ import annotations

from typing import Any, Dict, Tuple

_FONT_CACHE: Dict[Tuple[str, int, bool, bool], Any] = {}
_DEFAULT_FONT_CACHE: Dict[int, Any] = {}


def get_sys_font(
    pygame: Any,
    name: str = "Arial",
    size: int = 14,
    *,
    bold: bool = False,
    italic: bool = False,
) -> Any:
    """Return a cached ``pygame.font.SysFont`` instance."""
    key = (str(name), int(size), bool(bold), bool(italic))
    font = _FONT_CACHE.get(key)
    if font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont(key[0], key[1], bold=key[2], italic=key[3])
        _FONT_CACHE[key] = font
    return font


def clear_font_cache() -> None:
    """Clear cached fonts after a display/font subsystem reset."""
    _FONT_CACHE.clear()
    _DEFAULT_FONT_CACHE.clear()


def get_default_font(pygame: Any, size: int) -> Any:
    """Return a cached default pygame font."""
    key = int(size)
    font = _DEFAULT_FONT_CACHE.get(key)
    if font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(None, key)
        _DEFAULT_FONT_CACHE[key] = font
    return font
