"""Small pygame font cache for frame-rendered GUI overlays."""

from __future__ import annotations

from typing import Any, Dict, Tuple

_FONT_CACHE: Dict[Tuple[int, str, int, bool, bool], Any] = {}
_DEFAULT_FONT_CACHE: Dict[Tuple[int, int], Any] = {}


def get_sys_font(
    pygame: Any,
    name: str = "Arial",
    size: int = 14,
    *,
    bold: bool = False,
    italic: bool = False,
) -> Any:
    """Return a cached ``pygame.font.SysFont`` instance."""
    # A process may host more than one pygame-compatible rendering context
    # (tests, plugins, display reinitialization).  Font objects are owned by
    # their font subsystem and must not leak across those contexts.
    key = (id(pygame.font), str(name), int(size), bool(bold), bool(italic))
    font = _FONT_CACHE.get(key)
    if font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont(key[1], key[2], bold=key[3], italic=key[4])
        _FONT_CACHE[key] = font
    return font


def clear_font_cache() -> None:
    """Clear cached fonts after a display/font subsystem reset."""
    _FONT_CACHE.clear()
    _DEFAULT_FONT_CACHE.clear()


def get_default_font(pygame: Any, size: int) -> Any:
    """Return a cached default pygame font."""
    key = (id(pygame.font), int(size))
    font = _DEFAULT_FONT_CACHE.get(key)
    if font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(None, key[1])
        _DEFAULT_FONT_CACHE[key] = font
    return font
