from pathlib import Path

from src.gui.services.topology_export import export_topology


class DummyGui:
    def __init__(self, tmp_path: Path):
        self.maps = [type("M", (), {"graph": None, "room_positions": {}})()]
        self.current_map_idx = 0
        self.repo_root = tmp_path
        self.topology_export_dir = tmp_path / "exports" / "topology"
        self.topology_export_path = None
        self.messages = []

    def _set_message(self, msg, duration=0.0):
        self.messages.append((msg, duration))


class TinyGraph:
    def nodes(self):
        return [1, 2]

    def edges(self):
        return [(1, 2)]


def test_export_topology_no_graph_sets_message(tmp_path: Path):
    gui = DummyGui(tmp_path)
    export_topology(gui)
    assert gui.messages[-1][0] == "No topology graph available for this map"


def test_export_topology_writes_file_with_fallback(tmp_path: Path):
    gui = DummyGui(tmp_path)
    gui.maps[0].graph = TinyGraph()
    export_topology(gui)

    assert gui.topology_export_path is not None
    exported = Path(gui.topology_export_path)
    assert exported.exists()
    assert exported.suffix == ".dot"

