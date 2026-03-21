"""Auto-step and stop helpers for GUI auto-solve execution."""

from typing import Any


def stop_auto(gui: Any, reason: str, logger: Any) -> None:
    """Stop auto-solve mode with consistent logging and cleanup."""
    try:
        logger.debug("_stop_auto called: %s", reason)
        gui.auto_mode = False
        gui._auto_stuck_retries = 0
        if reason != "path complete":
            try:
                gui.auto_path = []
                gui.auto_step_idx = 0
            except Exception:
                pass
        try:
            if reason == "path complete":
                gui._set_message("Solution complete! Victory!")
            else:
                gui._set_message(f"Auto-solve stopped: {reason}")
        except Exception:
            pass
    except Exception:
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

    def _stop_auto_local(reason: str = None):
        try:
            logger.debug("_stop_auto_local calling _stop_auto: %s", reason)
            gui._stop_auto(reason)
        except Exception:
            logger.exception("_stop_auto_local failed: %s", reason)

    try:
        try:
            logger.debug(
                "_auto_step entry: auto_mode=%s auto_step_idx=%s path_len=%s",
                getattr(gui, "auto_mode", None),
                getattr(gui, "auto_step_idx", None),
                len(getattr(gui, "auto_path", []) if getattr(gui, "auto_path", None) else []),
            )
        except Exception:
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
            _stop_auto_local("path complete")
            gui._set_message("Solution complete!")
            gui.status_message = "Completed"
            return

        if gui.feature_flags.get("dstar_lite", False) and getattr(gui, "dstar_active", False) and getattr(gui, "dstar_solver", None):
            try:
                current_state = gui.env.get_state() if hasattr(gui.env, "get_state") else gui.env.state
                if gui.dstar_solver.needs_replan(current_state):
                    success, new_path, updated = gui.dstar_solver.replan(current_state)
                    if success and new_path:
                        curpos = gui.env.state.position
                        try:
                            idx = new_path.index(curpos)
                        except ValueError:
                            idx = 0
                        gui.auto_path = new_path
                        gui.auto_step_idx = idx
                        gui._set_message(f"D* Lite replanned ({updated} updates)")
            except Exception as e:
                logger.warning("D* Lite replanning failed: %s", e)

        if gui.env is None:
            gui._show_error("Environment not initialized")
            _stop_auto_local("env none")
            return

        if not hasattr(gui.env, "state") or gui.env.state is None:
            gui._show_error("Invalid environment state")
            _stop_auto_local("env.state invalid")
            return

        gui.auto_step_idx += 1
        target = gui.auto_path[gui.auto_step_idx]
        current = gui.env.state.position
        dr = target[0] - current[0]
        dc = target[1] - current[1]

        if getattr(gui, "inventory_needs_refresh", False):
            try:
                gui._update_inventory_and_hud()
            except Exception:
                pass
            finally:
                gui.inventory_needs_refresh = False

        if abs(dr) > 1 or abs(dc) > 1:
            old_state = game_state_cls(
                position=gui.env.state.position,
                keys=gui.env.state.keys,
                bomb_count=gui.env.state.bomb_count,
                has_boss_key=gui.env.state.has_boss_key,
                opened_doors=gui.env.state.opened_doors.copy() if hasattr(gui.env.state.opened_doors, "copy") else set(gui.env.state.opened_doors),
                collected_items=gui.env.state.collected_items.copy()
                if hasattr(gui.env.state.collected_items, "copy")
                else set(gui.env.state.collected_items),
            )

            gui.env.state.position = target
            gui._set_message(f"Teleport! {current} -> {target}")
            gui.status_message = "Teleporting..."

            try:
                gui._apply_pickup_at(target)
            except Exception as e:
                logger.warning("Pickup application failed on teleport: %s", e)

            try:
                gui._track_item_collection(old_state, gui.env.state)
                gui._track_item_usage(old_state, gui.env.state)
                try:
                    gui._sync_inventory_counters()
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Item tracking failed: %s", e)

            if gui.renderer:
                try:
                    gui.renderer.set_agent_position(target[0], target[1], immediate=True)
                except Exception as e:
                    logger.warning("Renderer update failed: %s", e)

            if gui.effects:
                try:
                    gui.effects.add_effect(ripple_effect_cls(target, (100, 200, 255)))
                except Exception as e:
                    logger.warning("Effect creation failed: %s", e)

            if target == gui.env.goal_pos:
                gui.env.won = True
                gui.env.done = True
                gui.auto_mode = False
                gui._set_message("AUTO-SOLVE: Victory!")
                gui.status_message = "Victory!"
            return

        old_state = game_state_cls(
            position=gui.env.state.position,
            keys=gui.env.state.keys,
            bomb_count=gui.env.state.bomb_count,
            has_boss_key=gui.env.state.has_boss_key,
            opened_doors=gui.env.state.opened_doors.copy() if hasattr(gui.env.state.opened_doors, "copy") else set(gui.env.state.opened_doors),
            collected_items=gui.env.state.collected_items.copy() if hasattr(gui.env.state.collected_items, "copy") else set(gui.env.state.collected_items),
        )

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
        except Exception as e:
            logger.warning("Block push check failed: %s", e)

        logger.debug(
            "_auto_step: performing env.step action=%r (int=%s) target=%s current=%s",
            action,
            int(action),
            target,
            current,
        )
        _state, _reward, done, info = gui.env.step(int(action))
        logger.debug(
            "_auto_step: env.step returned info=%r, new_pos=%s, env.keys=%s",
            info,
            getattr(gui.env.state, "position", None),
            getattr(gui.env.state, "keys", None),
        )

        new_pos = gui.env.state.position

        if not done and new_pos != target:
            retries = int(getattr(gui, "_auto_stuck_retries", 0)) + 1
            gui._auto_stuck_retries = retries
            logger.warning("_auto_step: blocked or desynced move (expected=%s, actual=%s, retry=%d)", target, new_pos, retries)

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
                _stop_auto_local("path blocked")
                gui.status_message = "Blocked"
            else:
                gui.status_message = "Retrying move..."
            return

        gui._auto_stuck_retries = 0
        gui.step_count += 1

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

        if done:
            gui.auto_mode = False
            if gui.env.won:
                gui._set_message("AUTO-SOLVE: Victory!")
                gui.status_message = "Victory!"
                if gui.effects:
                    try:
                        goal_pos = gui.env.goal_pos
                        gui.effects.add_effect(flash_effect_cls(goal_pos, (255, 215, 0), 0.5))
                    except Exception as e:
                        logger.warning("Victory effect failed: %s", e)
            else:
                gui._set_message(f"AUTO-SOLVE: Failed - {info.get('msg', '')}")
                gui.status_message = "Failed"

    except KeyError as e:
        gui._show_error(f"State access error: {str(e)}")
        _stop_auto_local("KeyError")
    except IndexError as e:
        gui._show_error(f"Path index error: {str(e)}")
        _stop_auto_local("IndexError")
    except AttributeError as e:
        logger.exception("Auto-step AttributeError caught: %s", e)
        gui._show_error(f"Invalid state attribute: {str(e)}")
        _stop_auto_local("AttributeError")
    except Exception as e:
        gui._show_error(f"Auto-solve error: {str(e)}")
        gui.auto_mode = False
        traceback_module.print_exc()