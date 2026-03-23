"""Adapter-backed map loading orchestration for GUI startup."""

from __future__ import annotations


def load_maps_from_adapter(*, os_module, file_path, print_fn=print):
    """Load all Zelda dungeon variants and optionally schedule async precalc."""
    try:
        from src.zelda_data.zelda_core import ZeldaDungeonAdapter, DungeonSolver
        from pathlib import Path

        data_root = Path(file_path).parent / "Data" / "The Legend of Zelda"

        if not data_root.exists():
            print_fn(f"Data folder not found: {data_root}")
            return None, None

        adapter = ZeldaDungeonAdapter(str(data_root))
        solver = DungeonSolver()

        maps = []
        map_names = []
        print_fn("Loading all 18 dungeon variants (9 dungeons x 2 variants)...")

        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                    stitched = adapter.stitch_dungeon(dungeon)

                    maps.append(stitched)

                    quest_name = "Quest 1" if variant == 1 else "Quest 2"
                    map_names.append(f"Dungeon {dungeon_num} ({quest_name})")

                    print_fn(f"  D{dungeon_num}-{variant}: Loaded - {stitched.global_grid.shape}")
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    print_fn(f"  D{dungeon_num}-{variant}: Error - {exc}")

        if os_module.environ.get("KLTN_PRECALC_SOLVES", "0") == "1":
            try:
                import threading

                def _precalc_worker():
                    print_fn("Starting background precalc solves for loaded maps...")
                    for idx, dungeon_map in enumerate(maps):
                        try:
                            result = solver.solve(dungeon_map)
                            status = "[OK]" if result.get("solvable") else "[X]"
                            print_fn(f"  [precalc] Map {idx + 1}: {status}")
                        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                            print_fn(f"  [precalc] Map {idx + 1}: Error - {exc}")

                threading.Thread(target=_precalc_worker, daemon=True).start()
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                print_fn("Precalc worker failed to start")

        return maps if maps else None, map_names if map_names else None

    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        print_fn(f"Error loading maps: {exc}")
        import traceback

        traceback.print_exc()
        return None, None


