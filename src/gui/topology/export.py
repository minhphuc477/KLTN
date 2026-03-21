"""Helpers for exporting current topology graph to DOT."""

from pathlib import Path
from typing import Any


def export_topology(gui: Any) -> None:
    """Export current map topology to a DOT file (if available)."""
    current = gui.maps[gui.current_map_idx]
    graph = getattr(current, "graph", None)
    room_positions = getattr(current, "room_positions", None)
    if graph is None:
        gui._set_message("No topology graph available for this map", 3.0)
        return

    try:
        import networkx as nx

        try:
            export_dir = Path(getattr(gui, "topology_export_dir", Path.cwd()))
            export_dir.mkdir(parents=True, exist_ok=True)
            fname = export_dir / f"topology_map_{gui.current_map_idx + 1}.dot"
            nx.nx_pydot.write_dot(graph, str(fname))

            if room_positions:
                with fname.open("a", encoding="utf-8") as f:
                    f.write("\n// room positions\n")
                    for room, (ry, rx) in room_positions.items():
                        f.write(f"// {room}: {ry},{rx}\n")

            try:
                display_path = fname.relative_to(getattr(gui, "repo_root", Path.cwd()))
            except Exception:
                display_path = fname
            gui._set_message(f"Topology exported to {display_path}")
            gui.topology_export_path = str(fname)
        except Exception as e:
            export_dir = Path(getattr(gui, "topology_export_dir", Path.cwd()))
            export_dir.mkdir(parents=True, exist_ok=True)
            fname = export_dir / f"topology_map_{gui.current_map_idx + 1}.dot"
            with fname.open("w", encoding="utf-8") as f:
                f.write("graph topology {\n")
                for n in graph.nodes():
                    f.write(f'  "{n}";\n')
                for u, v in graph.edges():
                    f.write(f'  "{u}" -- "{v}";\n')
                f.write("}\n")
            try:
                display_path = fname.relative_to(getattr(gui, "repo_root", Path.cwd()))
            except Exception:
                display_path = fname
            gui._set_message(f"Topology exported to {display_path} (manual)\n{e}")
            gui.topology_export_path = str(fname)
    except ImportError:
        gui._set_message("NetworkX not available - cannot export DOT automatically", 4.0)
