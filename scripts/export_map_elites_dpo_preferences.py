#!/usr/bin/env python
"""Export bounded same-condition room preferences from a trusted MAP-Elites archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.preference_buffer import QDPreferenceBuffer


def export(args: argparse.Namespace) -> Path:
    if not args.trust_pickle:
        raise ValueError("MAP-Elites archives are pickle files. Re-run with --trust-pickle only for a trusted local archive.")
    with args.archive.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("preference_buffer"), dict):
        raise ValueError(
            "Archive has no preference buffer. Enable generation.map_elites_preference_buffer_size "
            "and generate multiple seeds for each fixed mission graph."
        )
    state = payload["preference_buffer"]
    buffer = QDPreferenceBuffer(max_candidates=int(state.get("max_candidates", 0)))
    buffer.load_state_dict(state)
    return buffer.export_raw_pairs(args.output, min_score_margin=float(args.min_score_margin))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score-margin", type=float, default=0.05)
    parser.add_argument("--trust-pickle", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(export(parse_args()))
