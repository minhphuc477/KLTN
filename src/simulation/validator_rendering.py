"""Optional Pygame rendering helpers for :mod:`src.simulation.validator`."""

import logging
import os
from typing import Any

from src.core.definitions import ID_TO_NAME, SEMANTIC_PALETTE


logger = logging.getLogger("src.simulation.validator")


def init_render(env: Any) -> None:
    """Initialize Pygame rendering for a ZeldaLogicEnv-compatible object."""
    try:
        import pygame

        pygame.init()  # pylint: disable=no-member

        env.TILE_SIZE = 32
        screen_w = env.width * env.TILE_SIZE
        screen_h = env.height * env.TILE_SIZE + 60

        env._screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("ZAVE: Zelda Validation Environment")
        env._font = pygame.font.SysFont("Arial", 18, bold=True)

        env._load_images()
    except ImportError:
        print("Warning: Pygame not available. Rendering disabled.")
        env.render_mode = False


def load_images(env: Any) -> None:
    """Load tile images or create colored fallbacks."""
    import pygame

    tile_size = env.TILE_SIZE
    color_map = {
        SEMANTIC_PALETTE["VOID"]: (0, 0, 0),
        SEMANTIC_PALETTE["FLOOR"]: (200, 180, 140),
        SEMANTIC_PALETTE["WALL"]: (70, 70, 150),
        SEMANTIC_PALETTE["BLOCK"]: (139, 90, 43),
        SEMANTIC_PALETTE["DOOR_OPEN"]: (50, 50, 50),
        SEMANTIC_PALETTE["DOOR_LOCKED"]: (139, 69, 19),
        SEMANTIC_PALETTE["DOOR_BOMB"]: (100, 100, 100),
        SEMANTIC_PALETTE["DOOR_BOSS"]: (200, 50, 50),
        SEMANTIC_PALETTE["DOOR_PUZZLE"]: (150, 100, 200),
        SEMANTIC_PALETTE["ENEMY"]: (200, 50, 50),
        SEMANTIC_PALETTE["START"]: (100, 200, 100),
        SEMANTIC_PALETTE["TRIFORCE"]: (255, 215, 0),
        SEMANTIC_PALETTE["KEY_SMALL"]: (255, 200, 50),
        SEMANTIC_PALETTE["KEY_BOSS"]: (200, 100, 50),
        SEMANTIC_PALETTE["ELEMENT"]: (50, 50, 200),
        SEMANTIC_PALETTE["ELEMENT_FLOOR"]: (100, 100, 200),
    }

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    for tile_id, color in color_map.items():
        tile_name = ID_TO_NAME.get(tile_id, "unknown").lower()
        img_path = os.path.join(assets_dir, f"{tile_name}.png")

        if os.path.exists(img_path):
            try:
                img = pygame.image.load(img_path)
                env._images[tile_id] = pygame.transform.scale(
                    img, (tile_size, tile_size)
                )
                continue
            except (pygame.error, OSError, ValueError, TypeError) as exc:
                logger.debug(
                    "Could not load asset image %s: %s",
                    img_path,
                    exc,
                    exc_info=True,
                )

        surf = pygame.Surface((tile_size, tile_size))
        surf.fill(color)
        env._images[tile_id] = surf

    link_path = os.path.join(assets_dir, "link.png")
    if os.path.exists(link_path):
        try:
            img = pygame.image.load(link_path)
            env._link_img = pygame.transform.scale(img, (tile_size, tile_size))
        except (pygame.error, OSError, ValueError, TypeError) as exc:
            logger.debug(
                "Could not load link asset %s: %s",
                link_path,
                exc,
                exc_info=True,
            )


def render(env: Any) -> None:
    """Render the current environment state to its Pygame screen."""
    if not env.render_mode or env._screen is None:
        return

    import pygame

    env._screen.fill((30, 30, 30))

    for row in range(env.height):
        for col in range(env.width):
            tile_id = env.grid[row, col]
            image = env._images.get(
                tile_id, env._images.get(SEMANTIC_PALETTE["FLOOR"])
            )
            env._screen.blit(
                image, (col * env.TILE_SIZE, row * env.TILE_SIZE)
            )

    row, col = env.state.position
    x, y = col * env.TILE_SIZE, row * env.TILE_SIZE

    if env._link_img:
        env._screen.blit(env._link_img, (x, y))
    else:
        pygame.draw.rect(
            env._screen,
            (0, 255, 0),
            (x + 4, y + 4, env.TILE_SIZE - 8, env.TILE_SIZE - 8),
        )

    hud_y = env.height * env.TILE_SIZE
    pygame.draw.rect(
        env._screen,
        (0, 0, 0),
        (0, hud_y, env.width * env.TILE_SIZE, 60),
    )

    hud_text = (
        f"Keys: {env.state.keys} | Bombs: {env.state.bomb_count} | "
        f"Boss Key: {'Y' if env.state.has_boss_key else 'N'} | "
        f"Steps: {env.step_count}"
    )
    text_surf = env._font.render(hud_text, True, (255, 255, 255))
    env._screen.blit(text_surf, (10, hud_y + 10))

    status = "WON!" if env.won else ("DONE" if env.done else "Playing...")
    status_surf = env._font.render(
        status,
        True,
        (255, 255, 0) if env.won else (255, 255, 255),
    )
    env._screen.blit(status_surf, (10, hud_y + 35))

    pygame.display.flip()


def close(env: Any) -> None:
    """Release Pygame resources owned by the environment renderer."""
    if env.render_mode:
        try:
            import pygame

            pygame.quit()  # pylint: disable=no-member
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("Error during pygame.quit(): %s", exc, exc_info=True)


__all__ = ["close", "init_render", "load_images", "render"]
