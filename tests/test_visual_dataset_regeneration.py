"""Regression tests for visual dataset generation consistency.

Ensures that re-running extract_visual_levels.py produces identical grids
given the same templates and input images (deterministic CV guarantee).
"""
import numpy as np
import pytest
import os
import hashlib
from pathlib import Path
from src.data_processing.visual_extractor import extract_grid

DATA_ROOT = Path(__file__).parent.parent / 'Data' / 'The Legend of Zelda'


def _compute_grid_hash(arr: np.ndarray) -> str:
    """Stable hash of grid array (ids only, not confidence)."""
    ids = arr[:, :, 0].astype(np.int16)
    return hashlib.md5(ids.tobytes()).hexdigest()


def test_deterministic_extraction_smoke():
    """Verify extract_grid produces same output on repeated calls."""
    img = DATA_ROOT / 'Original' / 'tloz1_1.png'
    tileset = DATA_ROOT.parent.parent / 'assets' / 'Dungeon Tileset.png'
    
    if not img.exists():
        pytest.skip('Test image not available (tloz1_1.png missing)')
    if not tileset.exists():
        pytest.skip('Tileset not available (assets/Dungeon Tileset.png missing)')
    
    grid1 = extract_grid(str(img), str(tileset), tile_px=16)
    grid2 = extract_grid(str(img), str(tileset), tile_px=16)
    
    hash1 = _compute_grid_hash(grid1)
    hash2 = _compute_grid_hash(grid2)
    
    assert hash1 == hash2, "extract_grid must be deterministic"
    assert grid1.shape == (16, 11, 2), f"Expected (16,11,2), got {grid1.shape}"


def test_npz_metadata_roundtrip(tmp_path):
    """Verify NPZ metadata preservation and template hash stability."""
    import json
    
    # Create synthetic tileset + room
    from PIL import Image
    t0 = Image.new('RGBA', (16, 16), (200, 180, 140, 255))
    tileset = Image.new('RGBA', (16, 16))
    tileset.paste(t0, (0, 0))
    tileset_path = tmp_path / 'tileset.png'
    tileset.save(tileset_path)
    
    room = Image.new('RGBA', (11 * 16, 16 * 16))
    for r in range(16):
        for c in range(11):
            room.paste(t0, (c * 16, r * 16))
    room_path = tmp_path / 'room.png'
    room.save(room_path)
    
    # Extract and save NPZ
    arr = extract_grid(str(room_path), str(tileset_path), tile_px=16)
    template_hash = hashlib.md5(tileset_path.read_bytes()).hexdigest()[:8]
    metadata = {
        'template_hash': template_hash,
        'tile_size': 16,
        'source': 'test'
    }
    
    npz_path = tmp_path / 'room.npz'
    np.savez_compressed(npz_path, grid=arr, metadata=json.dumps(metadata))
    
    # Load and verify
    loaded = np.load(npz_path)
    loaded_meta = json.loads(str(loaded['metadata']))
    
    assert np.array_equal(loaded['grid'], arr), "Grid data must roundtrip exactly"
    assert loaded_meta['template_hash'] == template_hash, "Template hash must be preserved"
    assert loaded_meta['tile_size'] == 16


def test_template_hash_changes_on_modification(tmp_path):
    """Verify template hash detects tileset changes (regression safety)."""
    from PIL import Image
    
    tileset1 = Image.new('RGBA', (16, 16), (100, 100, 100, 255))
    path1 = tmp_path / 'tileset1.png'
    tileset1.save(path1)
    hash1 = hashlib.md5(path1.read_bytes()).hexdigest()[:8]
    
    tileset2 = Image.new('RGBA', (16, 16), (100, 100, 101, 255))  # 1px color diff
    path2 = tmp_path / 'tileset2.png'
    tileset2.save(path2)
    hash2 = hashlib.md5(path2.read_bytes()).hexdigest()[:8]
    
    assert hash1 != hash2, "Template hash must detect modifications"


def test_extraction_produces_valid_confidence_range():
    """Verify confidence values are in valid [0,1] range."""
    from PIL import Image
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Create synthetic tileset + room
        t0 = Image.new('RGBA', (16, 16), (200, 180, 140, 255))
        tileset = Image.new('RGBA', (32, 16))
        tileset.paste(t0, (0, 0))
        tileset.paste(t0, (16, 0))
        tileset_path = tmp / 'tileset.png'
        tileset.save(tileset_path)
        
        room = Image.new('RGBA', (11 * 16, 16 * 16))
        for r in range(16):
            for c in range(11):
                room.paste(t0, (c * 16, r * 16))
        room_path = tmp / 'room.png'
        room.save(room_path)
        
        arr = extract_grid(str(room_path), str(tileset_path), tile_px=16)
        conf = arr[:, :, 1]
        
        assert np.all(conf >= 0.0), "Confidence must be >= 0"
        assert np.all(conf <= 1.0), "Confidence must be <= 1"
        assert conf.mean() > 0.3, "Mean confidence should be reasonable for synthetic data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
