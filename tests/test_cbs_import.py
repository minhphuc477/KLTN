def test_import_cbs():
    """Simple import test to ensure CBS module is importable."""
    from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
    assert callable(CognitiveBoundedSearch)
