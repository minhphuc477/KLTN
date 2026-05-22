import json
from pathlib import Path

import numpy as np

from src.simulation.validator import Action, ZeldaLogicEnv


def test_real_checkpoint_new_quest_route_file_is_gui_loadable_and_replays():
    level_file = Path("examples/gui_demo_real_checkpoint_new_quest_level.txt")
    route_file = Path("examples/gui_demo_real_checkpoint_new_quest_route.json")
    metadata_file = Path("examples/gui_demo_real_checkpoint_new_quest_metadata.json")

    assert level_file.exists()
    assert route_file.exists()
    assert metadata_file.exists()

    grid = np.loadtxt(level_file, dtype=np.int32)
    route = json.loads(route_file.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    assert all(key in route for key in ("path", "start", "goal"))
    assert tuple(route["start"]) == tuple(int(v) for v in np.argwhere(grid == 21)[0])
    assert tuple(route["goal"]) == tuple(int(v) for v in np.argwhere(grid == 22)[0])
    assert metadata["validation"]["success"] is True

    path = [tuple(int(v) for v in point) for point in route["path"]]
    assert path[0] == tuple(route["start"])
    assert path[-1] == tuple(route["goal"])
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))

    action_for_delta = {
        (-1, 0): Action.UP,
        (1, 0): Action.DOWN,
        (0, -1): Action.LEFT,
        (0, 1): Action.RIGHT,
    }
    env = ZeldaLogicEnv(grid)
    assert env.state.position == path[0]
    for current, target in zip(path, path[1:]):
        delta = (target[0] - current[0], target[1] - current[1])
        env.step(action_for_delta[delta])
        assert env.state.position == target

    assert env.done is True
    assert env.won is True
