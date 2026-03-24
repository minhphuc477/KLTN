"""Convenience helper functions extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Dict


def test_all_dungeons(
    data_root: str,
    include_variants: bool,
    adapter_cls: Any,
    solver_cls: Any,
    logger: Any,
) -> Dict[str, Dict]:
    """Run solvability checks for all dungeons and variants."""
    adapter = adapter_cls(data_root)
    solver = solver_cls()
    results: Dict[str, Dict] = {}

    variants = [1, 2] if include_variants else [1]

    for dungeon_num in range(1, 10):
        for variant in variants:
            dungeon_key = f"D{dungeon_num}-{variant}" if include_variants else f"D{dungeon_num}"
            try:
                dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                stitched = adapter.stitch_dungeon(dungeon)
                result = solver.solve(stitched)
                results[dungeon_key] = result

                status = "SOLVABLE" if result["solvable"] else "NOT SOLVABLE"
                logger.info("%s: %s", dungeon_key, status)
                if result["solvable"]:
                    logger.debug(
                        "%s Path: %d steps, %d rooms",
                        dungeon_key,
                        result["path_length"],
                        result["rooms_traversed"],
                    )
                else:
                    logger.debug("%s Reason: %s", dungeon_key, result.get("reason", "Unknown"))
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                results[dungeon_key] = {"solvable": False, "error": str(exc)}
                logger.exception("%s: ERROR during processing", dungeon_key)

    solvable_count = sum(1 for value in results.values() if value.get("solvable"))
    total = len(results)
    logger.info("SUMMARY: %d/%d solvable (%.1f%%)", solvable_count, total, 100 * solvable_count / total)

    return results


def visualize_semantic_grid(grid, show_legend: bool, semantic_palette: Dict[str, int]) -> str:
    """Create an ASCII representation of a semantic tile grid."""
    symbol_map = {
        semantic_palette["VOID"]: " ",
        semantic_palette["FLOOR"]: ".",
        semantic_palette["WALL"]: "#",
        semantic_palette["BLOCK"]: "B",
        semantic_palette["DOOR_OPEN"]: "O",
        semantic_palette["DOOR_LOCKED"]: "L",
        semantic_palette["DOOR_BOMB"]: "X",
        semantic_palette["ENEMY"]: "E",
        semantic_palette["START"]: "S",
        semantic_palette["TRIFORCE"]: "T",
        semantic_palette["KEY"]: "k",
        semantic_palette["ITEM"]: "i",
        semantic_palette["ELEMENT"]: "~",
        semantic_palette["STAIR"]: "^",
        semantic_palette["BOSS"]: "B",
    }

    lines = []
    for row in grid:
        lines.append("".join(symbol_map.get(int(cell), "?") for cell in row))

    result = "\n".join(lines)

    if show_legend:
        result += "\n\nLegend: . floor, # wall, O open door, L locked door, X bomb wall"
        result += "\n        E enemy, S start, T triforce, k key, ~ hazard, ^ stair"

    return result
