from pathlib import Path

from src.gui.runtime.temp_file_tools import delete_files, find_temp_files, list_existing_paths


def test_list_existing_paths_deduplicates(tmp_path: Path):
    p = tmp_path / "a.tmp"
    p.write_text("x", encoding="utf-8")

    result = list_existing_paths([str(p), str(p), str(tmp_path / "missing.tmp")])
    assert result == [str(p)]


def test_find_temp_files_by_pattern(tmp_path: Path):
    a = tmp_path / "zave_solver_out_1.pkl"
    b = tmp_path / "zave_grid_1.npy"
    c = tmp_path / "other.txt"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    c.write_text("c", encoding="utf-8")

    found = find_temp_files(str(tmp_path), ["zave_solver_out_*.pkl", "zave_grid_*.npy"])
    assert str(a) in found
    assert str(b) in found
    assert str(c) not in found


def test_delete_files_reports_deleted(tmp_path: Path):
    p1 = tmp_path / "x.tmp"
    p2 = tmp_path / "y.tmp"
    p1.write_text("x", encoding="utf-8")
    p2.write_text("y", encoding="utf-8")

    deleted, failures = delete_files([str(p1), str(p2)])
    assert deleted == 2
    assert failures == []
    assert not p1.exists()
    assert not p2.exists()

