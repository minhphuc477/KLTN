"""Auto-step and stop helpers for GUI auto-solve execution."""

from collections import deque
from typing import Any, Optional

from src.core.definitions import SEMANTIC_PALETTE


def _snapshot_state(gui: Any, game_state_cls: Any) -> Any:
    """Create a detached snapshot of the current env state for delta tracking."""
    state = gui.env.state
    return game_state_cls(
        position=state.position,
        keys=state.keys,
        bomb_count=state.bomb_count,
        has_boss_key=state.has_boss_key,
        opened_doors=state.opened_doors.copy() if hasattr(state.opened_doors, "copy") else set(state.opened_doors),
        collected_items=state.collected_items.copy() if hasattr(state.collected_items, "copy") else set(state.collected_items),
    )


def _stop_auto_local(gui: Any, logger: Any, reason: str = None) -> None:
    """Best-effort wrapper that delegates to GUI-level stop handler."""
    try:
        logger.debug("_stop_auto_local calling _stop_auto: %s", reason)
        gui._stop_auto(reason)
    except (AttributeError, RuntimeError, ValueError, TypeError):  # pragma: no cover - defensive fallback
        logger.exception("_stop_auto_local failed: %s", reason)


def _refresh_inventory_if_needed(gui: Any) -> None:
    """Refresh inventory/HUD when a background worker flagged stale state."""
    if not getattr(gui, "inventory_needs_refresh", False):
        return
    try:
        gui._update_inventory_and_hud()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        # Keep auto-step alive even if a UI refresh fails.
        pass
    finally:
        gui.inventory_needs_refresh = False


def _is_cardinal_neighbor(current: tuple, target: tuple) -> bool:
    """Return True when target is one 4-directional move from current."""
    return abs(target[0] - current[0]) + abs(target[1] - current[1]) == 1


def _grid_shape(grid: Any) -> Optional[tuple]:
    if grid is None:
        return None
    shape = getattr(grid, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    try:
        return len(grid), len(grid[0])
    except (TypeError, IndexError):
        return None


def _grid_tile(grid: Any, pos: tuple) -> int:
    try:
        return int(grid[pos[0], pos[1]])
    except TypeError:
        return int(grid[pos[0]][pos[1]])


def _find_contiguous_replay_segment(gui: Any, current: tuple, target: tuple) -> Optional[list]:
    """Find a real 4-directional state-valid path to replace a non-adjacent replay step."""
    grid = getattr(getattr(gui, "env", None), "grid", None)
    shape = _grid_shape(grid)
    if not shape:
        return None

    height, width = shape
    if not (0 <= current[0] < height and 0 <= current[1] < width):
        return None
    if not (0 <= target[0] < height and 0 <= target[1] < width):
        return None

    if not hasattr(getattr(gui, "env", None), "try_move_pure"):
        try:
            from src.simulation.validator import BLOCKING_IDS, WATER_IDS

            blocked_ids = {int(v) for v in (BLOCKING_IDS | WATER_IDS)}
        except ImportError:
            blocked_ids = {
                int(SEMANTIC_PALETTE["VOID"]),
                int(SEMANTIC_PALETTE["WALL"]),
                int(SEMANTIC_PALETTE["BLOCK"]),
                int(SEMANTIC_PALETTE["ELEMENT"]),
            }

        queue = deque([current])
        parents = {current: None}
        while queue:
            pos = queue.popleft()
            if pos == target:
                path = []
                cursor = target
                while cursor is not None:
                    path.append(cursor)
                    cursor = parents[cursor]
                return list(reversed(path))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                npos = (pos[0] + dr, pos[1] + dc)
                if npos in parents:
                    continue
                if not (0 <= npos[0] < height and 0 <= npos[1] < width):
                    continue
                try:
                    if _grid_tile(grid, npos) in blocked_ids:
                        continue
                except (IndexError, TypeError, ValueError):
                    continue
                parents[npos] = pos
                queue.append(npos)
        return None

    start_state = getattr(getattr(gui, "env", None), "state", None)
    if start_state is None:
        return None

    try:
        start_state = start_state.copy()
        start_state.position = current
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None

    queue = deque([start_state])
    parents = {start_state: None}
    max_expansions = max(4096, min(30000, height * width * 16))
    expansions = 0
    while queue:
        state = queue.popleft()
        pos = state.position
        if pos == target:
            path = []
            cursor = state
            while cursor is not None:
                path.append(cursor.position)
                cursor = parents[cursor]
            return list(reversed(path))

        expansions += 1
        if expansions >= max_expansions:
            return None

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            npos = (pos[0] + dr, pos[1] + dc)
            if not (0 <= npos[0] < height and 0 <= npos[1] < width):
                continue
            try:
                can_move, next_state = gui.env.try_move_pure(state, npos, _grid_tile(grid, npos))
            except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
                continue
            if not can_move:
                continue
            if next_state in parents:
                continue
            parents[next_state] = state
            queue.append(next_state)

    return None


def _splice_contiguous_replay_segment(
    gui: Any,
    logger: Any,
    current: tuple,
    target: tuple,
) -> bool:
    """Replace the active non-adjacent path node with adjacent replay nodes."""
    segment = _find_contiguous_replay_segment(gui, current, target)
    if not segment or len(segment) < 2:
        return False

    idx = int(getattr(gui, "auto_step_idx", 0))
    path = list(getattr(gui, "auto_path", []) or [])
    gui.auto_path = path[:idx] + segment[1:] + path[idx + 1 :]
    logger.info(
        "Auto-solve replay expanded non-adjacent transition %s -> %s into %d grid steps",
        current,
        target,
        len(segment) - 1,
    )
    try:
        gui._set_message("Expanded solver shortcut into grid steps")
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass
    return True


def _maybe_replan_dstar(gui: Any, logger: Any) -> None:
    """Run one D* Lite replanning cycle when enabled."""
    if not gui.feature_flags.get("dstar_lite", False):
        return
    if not getattr(gui, "dstar_active", False) or not getattr(gui, "dstar_solver", None):
        return

    try:
        current_state = gui.env.get_state() if hasattr(gui.env, "get_state") else gui.env.state
        if not gui.dstar_solver.needs_replan(current_state):
            return

        success, new_path, updated = gui.dstar_solver.replan(current_state)
        if not success or not new_path:
            return

        curpos = gui.env.state.position
        try:
            idx = new_path.index(curpos)
        except ValueError:
            idx = 0

        gui.auto_path = new_path
        gui.auto_step_idx = idx
        gui._set_message(f"D* Lite replanned ({updated} updates)")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("D* Lite replanning failed: %s", exc)
        try:
            gui._set_message("D* Lite replanning failed; continuing with current path", 2.0)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass


def _validate_and_apply_teleport(
    gui: Any,
    logger: Any,
    current: Any,
    target: Any,
) -> bool:
    """Validate non-adjacent replay transition and apply state if allowed."""
    profile = str(getattr(gui.env, "rules_profile", "extended") or "extended").strip().lower()
    strict_original = profile in {"strict_original", "original", "nes"}
    vglc_strict = profile in {"vglc_strict", "vglc", "dataset"}

    curr_tile = int(gui.env.grid[current[0], current[1]])
    is_stair = curr_tile == SEMANTIC_PALETTE["STAIR"]
    is_door = curr_tile in {
        SEMANTIC_PALETTE["DOOR_OPEN"],
        SEMANTIC_PALETTE["DOOR_SOFT"],
        SEMANTIC_PALETTE["DOOR_LOCKED"],
        SEMANTIC_PALETTE["DOOR_BOMB"],
        SEMANTIC_PALETTE["DOOR_BOSS"],
    }
    if strict_original:
        can_teleport = is_stair
    elif vglc_strict:
        can_teleport = is_stair or is_door
    else:
        can_teleport = True

    allowed_targets = set()
    if can_teleport:
        try:
            from src.simulation.validator import StateSpaceAStar

            replay_solver = StateSpaceAStar(gui.env)
            if is_stair:
                allowed_targets.update(replay_solver.get_stair_destinations(current))
            if not strict_original:
                allowed_targets.update(
                    pos for pos, _cost, _edge_type in replay_solver.get_controlled_virtual_destinations(current, gui.env.state)
                )
                allowed_targets.update(
                    pos for pos, _cost, _edge_type in replay_solver.get_graph_warp_destinations(current, gui.env.state)
                )
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Teleport validation helper failed: %s", exc)

    if target not in allowed_targets:
        gui._show_error(f"Invalid teleport transition: {current} -> {target}")
        _stop_auto_local(gui, logger, "invalid teleport transition")
        return False

    target_tile = int(gui.env.grid[target[0], target[1]])
    can_move, new_state = gui.env.try_move_pure(gui.env.state, target, target_tile)
    if not can_move:
        gui._show_error(f"Teleport blocked by game rules: {current} -> {target}")
        _stop_auto_local(gui, logger, "teleport blocked")
        return False

    gui.env.state = new_state
    gui._set_message(f"Teleport! {current} -> {target}")
    gui.status_message = "Teleporting..."
    return True


def _conditional_door_ids() -> set:
    return {
        int(SEMANTIC_PALETTE["DOOR_LOCKED"]),
        int(SEMANTIC_PALETTE["DOOR_BOMB"]),
        int(SEMANTIC_PALETTE["DOOR_BOSS"]),
        int(SEMANTIC_PALETTE["DOOR_PUZZLE"]),
    }


def _recover_completed_door_interaction(
    gui: Any,
    logger: Any,
    old_state: Any,
    current: tuple,
    target: tuple,
    target_tile: int,
) -> bool:
    """Finish entering a door when replay consumed inventory but stayed in place."""
    state = getattr(getattr(gui, "env", None), "state", None)
    if state is None or getattr(state, "position", None) != current:
        return False

    target_tile_id = int(target_tile)
    opened_doors = set(getattr(state, "opened_doors", set()) or set())
    inventory_used = (
        int(getattr(state, "keys", 0)) < int(getattr(old_state, "keys", 0))
        or int(getattr(state, "bomb_count", 0)) < int(getattr(old_state, "bomb_count", 0))
        or (
            bool(getattr(old_state, "has_boss_key", False))
            and not bool(getattr(state, "has_boss_key", False))
        )
    )
    door_was_opened = target in opened_doors or target_tile_id == int(SEMANTIC_PALETTE["DOOR_OPEN"])
    if target_tile_id not in _conditional_door_ids() and not door_was_opened:
        return False
    if not inventory_used and not door_was_opened:
        return False

    open_id = int(SEMANTIC_PALETTE["DOOR_OPEN"])
    try:
        gui.env.grid[target[0], target[1]] = open_id
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass

    try:
        move_state = state.copy()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        move_state = state
    try:
        move_state.position = target
        if hasattr(move_state, "opened_doors"):
            move_state.opened_doors = set(getattr(move_state, "opened_doors", set()) or set()) | {target}
        gui.env.state = move_state
        logger.info(
            "Auto-solve replay recovered opened-door desync: %s -> %s (tile=%s)",
            current,
            target,
            target_tile_id,
        )
        gui.status_message = "Door entered"
        return True
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Door interaction recovery failed for %s -> %s: %s", current, target, exc)
        return False


def _post_move_updates(
    gui: Any,
    old_state: Any,
    new_pos: Any,
    flash_effect_cls: Any,
    logger: Any,
) -> None:
    """Apply item/HUD/renderer updates after successful movement."""
    gui._track_item_collection(old_state, gui.env.state)
    gui._track_item_usage(old_state, gui.env.state)

    if gui.modern_hud:
        gui.modern_hud.update_game_state(
            keys=gui.env.state.keys,
            bombs=gui.env.state.bomb_count,
            has_boss_key=gui.env.state.has_boss_key,
            position=new_pos,
            steps=gui.step_count,
            message=getattr(gui, "message", ""),
        )
        gui._sync_inventory_counters()
        if hasattr(gui.modern_hud, "keys_collected"):
            gui.modern_hud.keys_collected = gui.keys_collected
            gui.modern_hud.bombs_collected = gui.bombs_collected
            gui.modern_hud.boss_keys_collected = gui.boss_keys_collected
        if hasattr(gui.modern_hud, "keys_used"):
            gui.modern_hud.keys_used = getattr(gui, "keys_used", 0)
        if hasattr(gui.modern_hud, "bombs_used"):
            gui.modern_hud.bombs_used = getattr(gui, "bombs_used", 0)
        if hasattr(gui.modern_hud, "boss_keys_used"):
            gui.modern_hud.boss_keys_used = getattr(gui, "boss_keys_used", 0)

    if gui.renderer:
        gui.renderer.set_agent_position(new_pos[0], new_pos[1], immediate=False)

    if gui.env.done:
        gui.auto_mode = False
        if gui.env.won:
            gui._set_message("AUTO-SOLVE: Victory!")
            gui.status_message = "Victory!"
            if gui.effects:
                try:
                    gui.effects.add_effect(flash_effect_cls(gui.env.goal_pos, (255, 215, 0), 0.5))
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    logger.warning("Victory effect failed: %s", exc)
        else:
            gui._set_message("AUTO-SOLVE: Failed")
            gui.status_message = "Failed"


def stop_auto(gui: Any, reason: str, logger: Any) -> None:
    """Stop auto-solve mode with consistent logging and cleanup."""
    try:
        logger.debug("_stop_auto called: %s", reason)
        gui.auto_mode = False
        gui._auto_stuck_retries = 0
        if reason != "path complete":
            gui.auto_path = []
            gui.auto_step_idx = 0
        if reason == "path complete":
            gui._set_message("Solution complete! Victory!")
        else:
            gui._set_message(f"Auto-solve stopped: {reason}")
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.exception("_stop_auto failed: %s", reason)


def auto_step(
    gui: Any,
    logger: Any,
    game_state_cls: Any,
    action_enum: Any,
    ripple_effect_cls: Any,
    flash_effect_cls: Any,
    traceback_module: Any,
) -> None:
    """Execute one auto-solve step with guard rails and item/effect updates."""

    try:
        try:
            logger.debug(
                "_auto_step entry: auto_mode=%s auto_step_idx=%s path_len=%s",
                getattr(gui, "auto_mode", None),
                getattr(gui, "auto_step_idx", None),
                len(getattr(gui, "auto_path", []) if getattr(gui, "auto_path", None) else []),
            )
        except (AttributeError, TypeError, ValueError):
            logger.debug("_auto_step entry: failed to read entry state")

        if not gui.auto_mode:
            logger.debug("_auto_step: auto_mode disabled; returning")
            return

        if not hasattr(gui, "auto_path") or not gui.auto_path:
            logger.warning("_auto_step: No solution path available (path empty or missing)")
            gui._show_error("No solution path available")
            gui.auto_mode = False
            return

        if gui.auto_step_idx >= len(gui.auto_path) - 1:
            logger.info("_auto_step: Path complete at index %s/%s", gui.auto_step_idx, len(gui.auto_path) - 1)
            _stop_auto_local(gui, logger, "path complete")
            gui._set_message("Solution complete!")
            gui.status_message = "Completed"
            return

        _maybe_replan_dstar(gui, logger)

        if gui.env is None:
            gui._show_error("Environment not initialized")
            _stop_auto_local(gui, logger, "env none")
            return

        if not hasattr(gui.env, "state") or gui.env.state is None:
            gui._show_error("Invalid environment state")
            _stop_auto_local(gui, logger, "env.state invalid")
            return

        gui.auto_step_idx += 1
        target = gui.auto_path[gui.auto_step_idx]
        current = gui.env.state.position
        dr = target[0] - current[0]
        dc = target[1] - current[1]

        _refresh_inventory_if_needed(gui)

        if current == target:
            gui.status_message = "Realigning path..."
            return

        if not _is_cardinal_neighbor(current, target):
            if _splice_contiguous_replay_segment(gui, logger, current, target):
                target = gui.auto_path[gui.auto_step_idx]
                dr = target[0] - current[0]
                dc = target[1] - current[1]
            elif bool(getattr(gui, "feature_flags", {}).get("allow_replay_teleports", False)):
                old_state = _snapshot_state(gui, game_state_cls)
                if not _validate_and_apply_teleport(gui, logger, current, target):
                    return

                gui._track_item_collection(old_state, gui.env.state)
                gui._track_item_usage(old_state, gui.env.state)
                try:
                    gui._sync_inventory_counters()
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    pass

                if gui.renderer:
                    try:
                        gui.renderer.set_agent_position(target[0], target[1], immediate=True)
                    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                        logger.warning("Renderer update failed: %s", exc)

                if gui.effects:
                    try:
                        gui.effects.add_effect(ripple_effect_cls(target, (100, 200, 255)))
                    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                        logger.warning("Effect creation failed: %s", exc)

                if target == gui.env.goal_pos:
                    gui.env.won = True
                    gui.env.done = True
                    gui.auto_mode = False
                    gui._set_message("AUTO-SOLVE: Victory!")
                    gui.status_message = "Victory!"
                return
            else:
                gui._show_error(
                    f"Auto-solve path contains a non-adjacent transition without a grid route: {current} -> {target}"
                )
                _stop_auto_local(gui, logger, "non-adjacent replay transition")
                gui.status_message = "Replay path invalid"
                return

        if not _is_cardinal_neighbor(current, target):
            gui._show_error(f"Auto-solve path is not a 4-directional move: {current} -> {target}")
            _stop_auto_local(gui, logger, "invalid replay move")
            gui.status_message = "Replay path invalid"
            return

        old_state = _snapshot_state(gui, game_state_cls)

        if dr == -1:
            action = action_enum.UP
        elif dr == 1:
            action = action_enum.DOWN
        elif dc == -1:
            action = action_enum.LEFT
        else:
            action = action_enum.RIGHT

        try:
            gui._check_and_start_block_push(current, target, action)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Block push check failed: %s", exc)

        logger.debug(
            "_auto_step: performing env.step action=%r (int=%s) target=%s current=%s",
            action,
            int(action),
            target,
            current,
        )
        target_tile_before_step = _grid_tile(gui.env.grid, target)
        _state, _reward, done, _info = gui.env.step(int(action))
        logger.debug(
            "_auto_step: env.step returned info=%r, new_pos=%s, env.keys=%s",
            _info,
            getattr(gui.env.state, "position", None),
            getattr(gui.env.state, "keys", None),
        )

        new_pos = gui.env.state.position

        if not done and new_pos != target:
            if _recover_completed_door_interaction(
                gui,
                logger,
                old_state,
                current,
                target,
                target_tile_before_step,
            ):
                new_pos = gui.env.state.position
                done = bool(getattr(gui.env, "done", done))
            else:
                retries = int(getattr(gui, "_auto_stuck_retries", 0)) + 1
                gui._auto_stuck_retries = retries
                try:
                    current_tile = _grid_tile(gui.env.grid, current)
                    retry_target_tile = _grid_tile(gui.env.grid, target)
                except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
                    current_tile = "?"
                    retry_target_tile = "?"
                logger.warning(
                    "_auto_step: blocked or desynced move (expected=%s, actual=%s, retry=%d, target_tile=%s, current_tile=%s, info=%r)",
                    target,
                    new_pos,
                    retries,
                    retry_target_tile,
                    current_tile,
                    _info,
                )

                gui.auto_step_idx = max(0, gui.auto_step_idx - 1)

                if new_pos != current:
                    try:
                        realign_idx = gui.auto_path.index(new_pos, gui.auto_step_idx)
                        gui.auto_step_idx = realign_idx
                        gui._auto_stuck_retries = 0
                        logger.info("_auto_step: path realigned to index=%d at pos=%s", realign_idx, new_pos)
                    except ValueError:
                        logger.debug("_auto_step: could not realign path for pos=%s", new_pos)

                if retries >= 3:
                    gui._show_error("Auto-solve path blocked; stopping")
                    _stop_auto_local(gui, logger, "path blocked")
                    gui.status_message = "Blocked"
                else:
                    gui.status_message = "Retrying move..."
                return

        gui._auto_stuck_retries = 0
        gui.step_count += 1

        # Keep canonical done flag aligned for downstream UI helpers.
        gui.env.done = bool(done)
        _post_move_updates(gui, old_state, new_pos, flash_effect_cls, logger)

    except KeyError as e:
        gui._show_error(f"State access error: {str(e)}")
        _stop_auto_local(gui, logger, "KeyError")
    except IndexError as e:
        gui._show_error(f"Path index error: {str(e)}")
        _stop_auto_local(gui, logger, "IndexError")
    except AttributeError as e:
        logger.exception("Auto-step AttributeError caught: %s", e)
        gui._show_error(f"Invalid state attribute: {str(e)}")
        _stop_auto_local(gui, logger, "AttributeError")
    except (RuntimeError, ValueError, TypeError) as e:
        gui._show_error(f"Auto-solve error: {str(e)}")
        gui.auto_mode = False
        traceback_module.print_exc()
