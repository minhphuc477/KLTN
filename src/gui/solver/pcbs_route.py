"""Replay-safe route shaping for P-CBS traces used by the GUI."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple


Position = Tuple[int, int]


def _state_signature(state: Any) -> tuple:
    """Return the progression state that must match before a loop is removable."""
    return (
        tuple(state.position),
        int(getattr(state, "keys", 0)),
        int(getattr(state, "bomb_count", 0)),
        bool(getattr(state, "has_boss_key", False)),
        bool(getattr(state, "has_item", False)),
        frozenset(getattr(state, "opened_doors", set())),
        frozenset(getattr(state, "collected_items", set())),
        frozenset(getattr(state, "pushed_blocks", set())),
        frozenset(getattr(state, "defeated_enemies", set())),
        frozenset(getattr(state, "completed_puzzle_stages", set())),
        int(getattr(state, "current_floor", 0)),
    )


def _normalize_path(path: Iterable[Sequence[int]]) -> List[Position]:
    return [(int(point[0]), int(point[1])) for point in path]


def _replay_signatures(
    *,
    grid: Any,
    path: List[Position],
    solver_options: Any = None,
) -> List[tuple]:
    """Replay a path from scratch and capture the game-state signature per point."""
    from src.simulation.validator import Action, ZeldaLogicEnv

    if not path:
        return []

    action_for_delta = {
        (-1, 0): Action.UP,
        (1, 0): Action.DOWN,
        (0, -1): Action.LEFT,
        (0, 1): Action.RIGHT,
    }

    env_kwargs = {"render_mode": False}
    if solver_options is not None:
        env_kwargs["solver_options"] = solver_options
    env = ZeldaLogicEnv(grid, **env_kwargs)
    if tuple(env.state.position) != tuple(path[0]):
        raise ValueError(f"path starts at {path[0]}, but environment starts at {env.state.position}")

    signatures = [_state_signature(env.state)]
    for current, target in zip(path, path[1:]):
        delta = (int(target[0]) - int(current[0]), int(target[1]) - int(current[1]))
        action = action_for_delta.get(delta)
        if action is None:
            raise ValueError(f"path contains non-cardinal step {current} -> {target}")

        _obs, _reward, _done, _info = env.step(action)
        if tuple(env.state.position) != tuple(target):
            raise ValueError(f"path replay diverged at {current} -> {target}; now at {env.state.position}")
        signatures.append(_state_signature(env.state))

    return signatures


def _loop_erase_by_state(path: List[Position], signatures: List[tuple]) -> List[Position]:
    """Remove only loops that return to the exact same gameplay state."""
    output_path: List[Position] = []
    output_signatures: List[tuple] = []
    index_by_signature: dict[tuple, int] = {}

    for pos, signature in zip(path, signatures):
        if signature in index_by_signature:
            keep_idx = index_by_signature[signature]
            for removed_signature in output_signatures[keep_idx + 1:]:
                index_by_signature.pop(removed_signature, None)
            output_path = output_path[: keep_idx + 1]
            output_signatures = output_signatures[: keep_idx + 1]
            continue

        index_by_signature[signature] = len(output_path)
        output_path.append(pos)
        output_signatures.append(signature)

    return output_path


def compress_pcbs_route_for_replay(
    *,
    grid: Any,
    path: Iterable[Sequence[int]],
    solver_options: Any = None,
) -> tuple[List[Position], dict[str, Any]]:
    """
    Convert a raw P-CBS cognitive trace into a shorter GUI replay path.

    The compression is conservative: it removes a loop only when replaying the
    trace reaches the same position with the same inventory, opened doors,
    collected items, pushed blocks, defeated enemies, puzzle state, and floor.
    That keeps key, bomb, door, and item progression intact.
    """
    raw_path = _normalize_path(path)
    stats = {
        "raw_trajectory_len": len(raw_path),
        "display_path_len": len(raw_path),
        "loops_removed": 0,
        "compressed": False,
        "compression_error": None,
    }
    if len(raw_path) < 3:
        return raw_path, stats

    try:
        signatures = _replay_signatures(grid=grid, path=raw_path, solver_options=solver_options)
        compressed = _loop_erase_by_state(raw_path, signatures)
        # Verify the shortened replay reaches the same final progression state.
        compressed_signatures = _replay_signatures(
            grid=grid,
            path=compressed,
            solver_options=solver_options,
        )
        if not compressed_signatures or compressed_signatures[-1] != signatures[-1]:
            raise ValueError("compressed path does not preserve the final game state")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        stats["compression_error"] = str(exc)
        return raw_path, stats

    stats["display_path_len"] = len(compressed)
    stats["loops_removed"] = max(0, len(raw_path) - len(compressed))
    stats["compressed"] = bool(len(compressed) < len(raw_path))
    return compressed, stats
