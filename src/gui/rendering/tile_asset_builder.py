"""Tile and marker sprite builders extracted from gui_runner."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def default_tile_color_map(*, semantic_palette: dict) -> Dict[int, Tuple[int, int, int]]:
    """Return the default semantic tile color map used by GUI fallback assets."""
    return {
        semantic_palette["VOID"]: (20, 20, 20),
        semantic_palette["FLOOR"]: (200, 180, 140),
        semantic_palette["WALL"]: (60, 60, 140),
        semantic_palette["BLOCK"]: (139, 90, 43),
        semantic_palette["DOOR_OPEN"]: (100, 80, 60),
        semantic_palette["DOOR_LOCKED"]: (139, 69, 19),
        semantic_palette["DOOR_BOMB"]: (80, 80, 80),
        semantic_palette["DOOR_BOSS"]: (180, 40, 40),
        semantic_palette["DOOR_PUZZLE"]: (140, 80, 180),
        semantic_palette["DOOR_SOFT"]: (100, 100, 60),
        semantic_palette["ENEMY"]: (200, 50, 50),
        semantic_palette["START"]: (80, 180, 80),
        semantic_palette["TRIFORCE"]: (255, 215, 0),
        semantic_palette["BOSS"]: (150, 20, 20),
        semantic_palette["KEY_SMALL"]: (255, 200, 50),
        semantic_palette["KEY_BOSS"]: (200, 100, 50),
        semantic_palette["KEY_ITEM"]: (100, 200, 255),
        semantic_palette["ITEM_MINOR"]: (200, 200, 200),
        semantic_palette["ELEMENT"]: (50, 80, 180),
        semantic_palette["ELEMENT_FLOOR"]: (80, 100, 160),
        semantic_palette["STAIR"]: (120, 100, 80),
        semantic_palette["PUZZLE"]: (180, 100, 180),
    }


def _draw_special_tile_details(*, surface: Any, tile_id: int, tile_size: int, semantic_palette: dict, pygame: Any) -> None:
    """Draw per-tile decorative markers for important semantics."""
    if tile_id == semantic_palette["DOOR_LOCKED"]:
        pygame.draw.circle(surface, (255, 200, 50), (tile_size // 2, tile_size // 2 - 4), 4)
        pygame.draw.rect(surface, (255, 200, 50), (tile_size // 2 - 2, tile_size // 2, 4, 8))
    elif tile_id == semantic_palette["DOOR_BOMB"]:
        pygame.draw.line(surface, (40, 40, 40), (8, 8), (24, 24), 2)
        pygame.draw.line(surface, (40, 40, 40), (24, 8), (8, 24), 2)
    elif tile_id == semantic_palette["KEY_SMALL"]:
        pygame.draw.circle(surface, (255, 255, 100), (16, 10), 9)
        pygame.draw.circle(surface, (255, 215, 0), (16, 10), 6)
        pygame.draw.rect(surface, (255, 215, 0), (14, 10, 4, 16))
        pygame.draw.rect(surface, (255, 215, 0), (14, 22, 2, 3))
        pygame.draw.rect(surface, (255, 215, 0), (16, 24, 2, 2))
        pygame.draw.circle(surface, (255, 255, 200), (17, 9), 2)
    elif tile_id == semantic_palette["TRIFORCE"]:
        points = [(16, 4), (4, 28), (28, 28)]
        pygame.draw.polygon(surface, (255, 255, 200), points)
        pygame.draw.polygon(surface, (200, 180, 0), points, 2)
    elif tile_id == semantic_palette["ENEMY"]:
        pygame.draw.circle(surface, (255, 100, 100), (16, 16), 10)
        pygame.draw.circle(surface, (0, 0, 0), (12, 12), 3)
        pygame.draw.circle(surface, (0, 0, 0), (20, 12), 3)
    elif tile_id == semantic_palette["START"]:
        pygame.draw.rect(surface, (60, 140, 60), (4, 4, 24, 24))
        for i in range(4):
            pygame.draw.line(surface, (40, 100, 40), (8, 8 + i * 6), (24, 8 + i * 6), 2)
    elif tile_id == semantic_palette["STAIR"]:
        for i in range(4):
            pygame.draw.rect(surface, (100, 80, 60), (4 + i * 4, 20 - i * 4, 20 - i * 4, 4))
    elif tile_id == semantic_palette["WALL"]:
        pygame.draw.rect(surface, (50, 50, 120), (2, 2, 28, 28), 2)
        pygame.draw.line(surface, (70, 70, 150), (0, 16), (32, 16), 1)
        pygame.draw.line(surface, (70, 70, 150), (16, 0), (16, 32), 1)
    elif tile_id == semantic_palette["BLOCK"]:
        pygame.draw.rect(surface, (100, 60, 30), (2, 2, 28, 28), 2)
    elif tile_id == semantic_palette["DOOR_OPEN"]:
        pygame.draw.rect(surface, (40, 30, 20), (8, 0, 16, 32))
    elif tile_id == semantic_palette["ELEMENT"]:
        for i in range(4):
            pygame.draw.arc(surface, (80, 120, 200), (i * 8, 8, 16, 16), 0, 3.14, 2)
            pygame.draw.arc(surface, (80, 120, 200), (i * 8, 16, 16, 16), 3.14, 6.28, 2)


def build_tile_images(*, tile_size: int, color_map: Dict[int, Tuple[int, int, int]], semantic_palette: dict, pygame: Any) -> Dict[int, Any]:
    """Build fallback tile surfaces for each semantic tile id."""
    images: Dict[int, Any] = {}
    for tile_id, color in color_map.items():
        surface = pygame.Surface((tile_size, tile_size))
        surface.fill(color)
        _draw_special_tile_details(
            surface=surface,
            tile_id=tile_id,
            tile_size=tile_size,
            semantic_palette=semantic_palette,
            pygame=pygame,
        )
        try:
            images[tile_id] = surface.convert_alpha()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            images[tile_id] = surface
    return images


def build_stair_marker_sprite(*, tile_size: int, pygame: Any) -> tuple[Any, float]:
    """Build stair marker sprite and return (sprite, initial_phase)."""
    sprite = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    sprite.fill((0, 0, 0, 0))

    pygame.draw.rect(sprite, (255, 220, 100, 180), (0, 0, tile_size, tile_size))
    pygame.draw.rect(sprite, (255, 200, 50), (1, 1, tile_size - 2, tile_size - 2), 4)

    points = [
        (tile_size // 2, tile_size // 6),
        (tile_size // 6, tile_size * 5 // 6),
        (tile_size * 5 // 6, tile_size * 5 // 6),
    ]
    pygame.draw.polygon(sprite, (255, 245, 180), points)
    pygame.draw.polygon(sprite, (255, 200, 50), points, 2)
    pygame.draw.circle(sprite, (255, 255, 220, 64), (tile_size // 2, tile_size // 2), max(6, tile_size // 6))

    try:
        sprite = sprite.convert_alpha()
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        pass
    return sprite, 0.0

