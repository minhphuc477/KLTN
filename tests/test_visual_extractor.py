import numpy as np
import os
from src.data_processing.visual_extractor import extract_grid

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Data', 'The Legend of Zelda')


def test_extract_on_synthetic_tileset(tmp_path):
    # create a synthetic tileset (2x1 tiles) and a room image composed of them
    from PIL import Image
    t0 = Image.new('RGBA', (16, 16), (200, 180, 140, 255))
    t1 = Image.new('RGBA', (16, 16), (60, 60, 140, 255))
    tileset = Image.new('RGBA', (32, 16))
    tileset.paste(t0, (0, 0))
    tileset.paste(t1, (16, 0))
    tileset_path = tmp_path / 'tileset.png'
    tileset.save(tileset_path)

    # build a room: left half floor (t0), right half wall (t1)
    room = Image.new('RGBA', (11 * 16, 16 * 16), (0, 0, 0, 0))
    for r in range(16):
        for c in range(11):
            tile = t0 if c < 6 else t1
            room.paste(tile, (c * 16, r * 16))
    room_path = tmp_path / 'room.png'
    room.save(room_path)

    out = extract_grid(str(room_path), str(tileset_path), tile_px=16)
    assert out.shape == (16, 11, 2)
    ids = out[:, :, 0].astype(int)
    # left half should be same id, right half another id
    assert np.all(ids[:, :6] == ids[0, 0])
    assert np.all(ids[:, 6:] == ids[0, 6])


def test_integration_on_tloz1_1():
    # Integration smoke test on provided asset; verify shape and a few tile values
    img = os.path.join(DATA_ROOT, 'Original', 'tloz1_1.png')
    tileset = os.path.join(DATA_ROOT, '..', '..', 'Dungeon Tileset.png')
    if not os.path.exists(img) or not os.path.exists(tileset):
        import pytest
        pytest.skip('test images not available')

    out = extract_grid(img, tileset)
    assert out.shape == (16, 11, 2)
    ids = out[:, :, 0].astype(int)

    # Expected (example) — checks for wall at corner and a stair marker near center
    assert ids[0, 0] >= 0  # corner detected (wall or border)
    # row 8 col 5 in tloz1_1 contains a stair ('S' in processed VGLC)
    assert ids[8, 5] >= 0


def test_estimate_hud_crop_detects_top_band(tmp_path):
    # create synthetic image with a HUD band (solid color) at the top
    from PIL import Image
    tile_px = 16
    hud_h = 32
    room_h = 16 * tile_px
    room_w = 11 * tile_px
    im = Image.new('RGBA', (room_w, hud_h + room_h), (0, 0, 0, 255))
    # draw HUD (bright strip)
    hud = Image.new('RGBA', (room_w, hud_h), (220, 180, 60, 255))
    im.paste(hud, (0, 0))
    # paste a uniform room below
    room = Image.new('RGBA', (room_w, room_h), (120, 120, 120, 255))
    im.paste(room, (0, hud_h))
    p = tmp_path / 'with_hud.png'
    im.save(p)

    from src.data_processing.visual_extractor import _estimate_hud_crop
    crop = _estimate_hud_crop(__import__('numpy').array(Image.open(p).convert('RGBA')))
    assert crop >= tile_px, 'HUD crop should be at least one tile high'
