import pytest
from gui_runner import ZeldaGUI


def test_solver_comparison_runs():
    g = ZeldaGUI()
    g._run_solver_comparison()
    assert g.solver_comparison_results is not None
    assert isinstance(g.solver_comparison_results, list)
    # At least A*, BFS should be present
    names = [r['name'] for r in g.solver_comparison_results]
    assert 'A*' in names
    assert 'BFS' in names
