from types import SimpleNamespace

from src.gui.controls.map_elites_controls import map_elites_worker, start_map_elites


class _Thread:
    def __init__(self, target=None, args=None, daemon=True):
        self.target = target
        self.args = args or ()
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started


class _Threading:
    Thread = _Thread


class _Logger:
    def __init__(self):
        self.exc = []

    def exception(self, *args, **kwargs):
        self.exc.append((args, kwargs))


def test_start_map_elites_starts_background_thread():
    toasts = []
    gui = SimpleNamespace(
        map_elites_thread=None,
        maps=[[[0]]],
        _show_toast=lambda msg, duration, toast_type="info": toasts.append((msg, duration, toast_type)),
        _map_elites_worker=lambda *_args: None,
    )

    start_map_elites(gui=gui, n_samples=10, resolution=8, threading_module=_Threading)

    assert gui.map_elites_thread is not None
    assert gui.map_elites_thread.started is True
    assert toasts and "Starting MAP-Elites" in toasts[0][0]


def test_map_elites_worker_stores_result_and_heatmap_path(tmp_path):
    toasts = []
    gui = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        map_elites_result=None,
        map_elites_heatmap_path=None,
        _show_toast=lambda msg, duration, toast_type="info": toasts.append((msg, duration, toast_type)),
    )

    class _Occ:
        pass

    def _run(_maps, resolution=20):
        return ("evaluator", _Occ())

    out_paths = []

    def _plot(_occ, out_path):
        out_paths.append(out_path)

    map_elites_worker(
        gui=gui,
        maps=[[[0]]],
        n_samples=5,
        resolution=12,
        logger=_Logger(),
        os_module=__import__("os"),
        run_map_elites_on_maps_fn=_run,
        plot_heatmap_fn=_plot,
    )

    assert gui.map_elites_result == "evaluator"
    assert gui.map_elites_heatmap_path is not None
    assert out_paths and out_paths[0].endswith("map_elites_heatmap_res12.png")
    assert toasts[-1][2] == "success"

