import json
import os
from scripts.curate_tileset import curate


def test_curate_runs_and_outputs(tmp_path):
    source = os.path.join("Data", "Assets", "NES - The Legend of Zelda - Tilesets - Dungeon Tileset.png")
    out = tmp_path / "out"
    out_str = str(out)
    out.mkdir()
    res = curate(source, out_str, tile_size=8, top_n=None, min_freq=1, k=16)
    # check files
    tiles_file = out / "tiles_curation.json"
    groups_file = out / "groups.json"
    atlas_file = out / "atlas_plan.json"
    assert tiles_file.exists()
    assert groups_file.exists()
    assert atlas_file.exists()

    tiles = json.load(open(tiles_file))
    atlas = json.load(open(atlas_file))

    total_tiles = res["n_base_tiles"]
    selected = atlas["ordered_uids"]

    # coverage: sum freq of selected uids
    freq_sum = 0
    for uid in selected:
        freq_sum += tiles["tiles"][uid]["freq"]

    coverage = freq_sum / total_tiles
    assert coverage >= 0.9, f"Coverage too low: {coverage}"
