"""Link sprite builder extracted from gui_runner."""

from __future__ import annotations

from typing import Any


def build_link_sprite(*, tile_size: int, pygame: Any) -> Any:
    """Create the fallback detailed Link sprite."""
    link_img = pygame.Surface((tile_size - 4, tile_size - 4), pygame.SRCALPHA)
    link_img.fill((0, 0, 0, 0))

    green = (0, 168, 0)
    skin = (252, 216, 168)
    brown = (136, 112, 0)
    dark_green = (0, 120, 0)

    pygame.draw.rect(link_img, green, (8, 12, 12, 12))
    pygame.draw.rect(link_img, dark_green, (6, 18, 4, 8))
    pygame.draw.rect(link_img, dark_green, (18, 18, 4, 8))

    pygame.draw.rect(link_img, skin, (8, 2, 12, 10))
    pygame.draw.circle(link_img, (0, 0, 0), (11, 6), 2)
    pygame.draw.circle(link_img, (0, 0, 0), (17, 6), 2)

    pygame.draw.rect(link_img, brown, (6, 0, 16, 4))
    pygame.draw.rect(link_img, brown, (4, 2, 4, 6))
    pygame.draw.rect(link_img, brown, (20, 2, 4, 6))

    pygame.draw.rect(link_img, brown, (2, 14, 6, 10))
    pygame.draw.rect(link_img, (200, 150, 50), (3, 15, 4, 8))

    pygame.draw.rect(link_img, (180, 180, 180), (22, 12, 4, 14))
    pygame.draw.rect(link_img, brown, (22, 10, 4, 4))

    try:
        return link_img.convert_alpha()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return link_img

