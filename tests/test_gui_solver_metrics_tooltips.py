from src.gui.services.solver_metrics_tooltips import format_cbs_metrics_tooltip


def test_format_cbs_metrics_tooltip_contains_expected_lines():
    metrics = {
        "confusion_index": 0.1234,
        "navigation_entropy": 0.4567,
        "cognitive_load": 0.789,
        "aha_latency": 12,
        "unique_tiles": 34,
        "peak_memory": 2,
        "replans": 5,
        "confusion_events": 1,
    }

    text = format_cbs_metrics_tooltip(metrics)

    assert "Confusion Index: 0.123" in text
    assert "Navigation Entropy: 0.457" in text
    assert "Cognitive Load: 0.789" in text
    assert "Aha Latency: 12 steps" in text
    assert "Unique Tiles: 34" in text
    assert "Peak Memory: 2 items" in text
    assert "Replans: 5" in text
    assert "Confusion Events: 1" in text

