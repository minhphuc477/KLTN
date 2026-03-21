import os
import pickle

from src.gui.services.solver_launching import (
    create_solver_temp_files,
    launch_solver_process,
    solver_thread_fallback_worker,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class DummyGui:
    def __init__(self):
        self.solver_proc = None
        self.solver_thread = object()
        self.solver_starting = True
        self.solver_running = True
        self.solver_start_time = 123.0
        self.messages = []

    def _set_message(self, msg):
        self.messages.append(msg)


def test_create_solver_temp_files_creates_paths():
    out_file, grid_file = create_solver_temp_files([[1, 2], [3, 4]])

    assert isinstance(out_file, str)
    assert os.path.dirname(out_file)
    # output file path is reserved but may be removed until worker writes.
    assert out_file.endswith(".pkl")

    if grid_file is not None:
        assert os.path.exists(grid_file)
        try:
            os.remove(grid_file)
        except Exception:
            pass


class _DummyProcess:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pid = 4321
        self._alive = False

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive


class _DummyMultiprocessing:
    Process = _DummyProcess


def test_launch_solver_process_updates_gui_state():
    gui = DummyGui()

    def _run_solver_and_dump(*args, **kwargs):
        return None

    launch_kwargs = {
        "grid_arr": [[0]],
        "grid_file": None,
        "start": (0, 0),
        "goal": (0, 1),
        "alg_idx": 0,
        "flags": {},
        "priority_options": {},
        "out_file": "dummy.pkl",
        "graph": None,
        "room_to_node": None,
        "room_positions": None,
        "node_to_room": None,
    }

    launch_solver_process(
        gui=gui,
        launch_kwargs=launch_kwargs,
        run_solver_and_dump=_run_solver_and_dump,
        multiprocessing_module=_DummyMultiprocessing,
        logger=DummyLogger(),
    )

    assert gui.solver_proc is not None
    assert gui.solver_proc.is_alive() is True
    assert gui.solver_thread is None
    assert gui.solver_starting is False
    assert gui.messages[-1] == "Solver started in background"


def test_solver_thread_fallback_worker_writes_output_and_resets_state(tmp_path):
    gui = DummyGui()
    out_file = tmp_path / "solver_out.pkl"

    def _solve_in_subprocess(*args, **kwargs):
        return {"success": True, "path": [(0, 0), (0, 1)]}

    launch_kwargs = {
        "grid_arr": [[0]],
        "start": (0, 0),
        "goal": (0, 1),
        "alg_idx": 0,
        "flags": {},
        "priority_options": {},
        "out_file": str(out_file),
        "graph": None,
        "room_to_node": None,
        "room_positions": None,
        "node_to_room": None,
    }

    solver_thread_fallback_worker(
        gui=gui,
        launch_kwargs=launch_kwargs,
        solve_in_subprocess=_solve_in_subprocess,
        logger=DummyLogger(),
    )

    assert gui.solver_running is False
    assert gui.solver_start_time is None
    assert gui.solver_starting is False
    assert gui.solver_thread is None
    assert out_file.exists()

    payload = pickle.loads(out_file.read_bytes())
    assert payload["success"] is True

