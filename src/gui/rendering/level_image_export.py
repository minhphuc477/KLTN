"""Small semantic-grid PNG exporter used by GUI TXT exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_TILE_COLORS = {
    0: (14, 17, 22),
    1: (204, 184, 139),
    2: (58, 70, 140),
    3: (130, 88, 48),
    10: (72, 50, 33),
    11: (138, 76, 22),
    12: (78, 78, 82),
    13: (135, 78, 174),
    14: (180, 42, 42),
    15: (110, 108, 65),
    20: (214, 62, 60),
    21: (74, 171, 92),
    22: (255, 218, 64),
    23: (142, 24, 34),
    30: (255, 202, 62),
    31: (220, 124, 58),
    32: (95, 197, 235),
    33: (225, 225, 225),
    40: (45, 78, 164),
    41: (78, 104, 164),
    42: (126, 98, 74),
    43: (180, 108, 180),
}

_MARKERS = {
    11: "K",
    12: "B",
    13: "P",
    14: "BK",
    15: "S",
    20: "E",
    21: "ST",
    22: "GO",
    23: "BO",
    30: "KY",
    31: "BK",
    32: "IT",
    33: "I",
    42: "S",
    43: "PZ",
}


def save_level_grid_png(
    grid: Any,
    path: str | Path,
    *,
    np_module: Any = None,
    tile_px: int = 16,
    room_width: int = 11,
    room_height: int = 16,
) -> Path:
    """Render a semantic integer grid as a readable PNG and return the path."""
    np_module = np_module or __import__("numpy")
    from PIL import Image, ImageDraw, ImageFont

    array = np_module.asarray(grid, dtype=np_module.int32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D grid for image export, got shape {array.shape}")

    tile_px = max(4, int(tile_px))
    height, width = array.shape
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (width * tile_px, height * tile_px), _TILE_COLORS[0])
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(8, tile_px // 2))
    except (OSError, ValueError):
        font = ImageFont.load_default()

    for row in range(height):
        y0 = row * tile_px
        for col in range(width):
            x0 = col * tile_px
            tile = int(array[row, col])
            color = _TILE_COLORS.get(tile, (230, 0, 230))
            draw.rectangle((x0, y0, x0 + tile_px - 1, y0 + tile_px - 1), fill=color)
            if tile == 2 and tile_px >= 10:
                draw.rectangle((x0 + 1, y0 + 1, x0 + tile_px - 2, y0 + tile_px - 2), outline=(44, 49, 105))
                draw.line((x0, y0 + tile_px // 2, x0 + tile_px - 1, y0 + tile_px // 2), fill=(76, 88, 161))
            elif tile in {10, 11, 12, 13, 14, 15} and tile_px >= 10:
                draw.rectangle(
                    (x0 + tile_px // 4, y0, x0 + (3 * tile_px) // 4, y0 + tile_px - 1),
                    fill=(34, 24, 18),
                )

            marker = _MARKERS.get(tile)
            if marker and tile_px >= 14:
                bbox = draw.textbbox((0, 0), marker, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w <= tile_px - 2:
                    tx = x0 + (tile_px - text_w) / 2
                    ty = y0 + (tile_px - text_h) / 2 - 1
                    draw.text((tx + 1, ty + 1), marker, fill=(0, 0, 0), font=font)
                    draw.text((tx, ty), marker, fill=(255, 255, 245), font=font)

    grid_line = (0, 0, 0)
    room_line = (245, 245, 245)
    for col in range(width + 1):
        x = col * tile_px
        is_room = int(room_width) > 0 and col % int(room_width) == 0
        draw.line((x, 0, x, height * tile_px), fill=room_line if is_room else grid_line, width=2 if is_room else 1)
    for row in range(height + 1):
        y = row * tile_px
        is_room = int(room_height) > 0 and row % int(room_height) == 0
        draw.line((0, y, width * tile_px, y), fill=room_line if is_room else grid_line, width=2 if is_room else 1)

    image.save(out_path)
    return out_path
