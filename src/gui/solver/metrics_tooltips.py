"""Formatting helpers for solver metrics tooltip strings."""


def format_cbs_metrics_tooltip(cbs_metrics: dict) -> str:
    """Format CBS metrics dictionary into a multi-line tooltip string."""
    lines = [
        f"Confusion Index: {cbs_metrics['confusion_index']:.3f}",
        f"Navigation Entropy: {cbs_metrics['navigation_entropy']:.3f}",
        f"Cognitive Load: {cbs_metrics['cognitive_load']:.3f}",
        f"Aha Latency: {cbs_metrics['aha_latency']} steps",
        f"Unique Tiles: {cbs_metrics['unique_tiles']}",
        f"Peak Memory: {cbs_metrics['peak_memory']} items",
        f"Replans: {cbs_metrics['replans']}",
        f"Confusion Events: {cbs_metrics['confusion_events']}",
    ]
    return "\n".join(lines)
