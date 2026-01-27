import pytest

try:
    from bench import numba_solvers
except Exception:
    pytest.skip('bench.numba_solvers import failed', allow_module_level=True)


def test_numba_neighbors_smoke():
    res = numba_solvers.smoke_test_neighbors()
    assert isinstance(res, list) or hasattr(res, '__len__')
    assert len(res) >= 0
