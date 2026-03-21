from src.gui.runtime.temp_file_management import (
    collect_temp_file_candidates,
    delete_temp_files,
    open_temp_folder,
)


class DummyGui:
    def __init__(self):
        self.messages = []
        self.solver_outfile = "solver.pkl"
        self.solver_gridfile = "grid.npy"
        self.preview_outfile = "preview.pkl"
        self.preview_gridfile = "preview.npy"
        self.solver_running = False
        self.preview_proc = None

    def _set_message(self, msg, duration=0.0):
        self.messages.append((msg, duration))


class DummyTempfile:
    @staticmethod
    def gettempdir():
        return "TMP"


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_open_temp_folder_success_sets_message():
    gui = DummyGui()
    open_temp_folder(gui, DummyTempfile, lambda p: (True, ""))
    assert gui.messages[-1][0] == "Opened temp folder: TMP"


def test_collect_temp_file_candidates_calls_helpers():
    gui = DummyGui()

    tracked, stale = collect_temp_file_candidates(
        gui,
        DummyTempfile,
        list_existing_paths_helper=lambda x: [p for p in x if p],
        find_temp_files_helper=lambda d, pats: [f"{d}:{len(pats)}"],
    )

    assert "solver.pkl" in tracked
    assert stale == ["TMP:7"]


def test_delete_temp_files_deletes_and_clears_handles():
    gui = DummyGui()
    deleted_inputs = []

    class DummyOs:
        class path:
            @staticmethod
            def exists(_):
                return False

    def collect_candidates():
        return ["solver.pkl", "grid.npy"], ["extra.pkl"]

    def delete_helper(paths):
        deleted_inputs.extend(paths)
        return (len(paths), [])

    delete_temp_files(
        gui,
        DummyOs,
        DummyLogger(),
        collect_candidates,
        list_existing_paths_helper=lambda p: p,
        delete_files_helper=delete_helper,
    )

    assert deleted_inputs == ["solver.pkl", "grid.npy", "extra.pkl"]
    assert gui.solver_outfile is None
    assert gui.solver_gridfile is None
    assert gui.preview_outfile is None
    assert gui.preview_gridfile is None
    assert gui.messages[-1][0].startswith("Deleted 3 temp files")

