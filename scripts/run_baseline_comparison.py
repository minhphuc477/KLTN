"""Compatibility entrypoint for baseline-comparison reports.

This wraps ``scripts.compare_protocol_to_baselines`` under the audit-requested
script name. It compares an already-generated fixed-graph protocol summary
against matched-budget topology and PCG Benchmark alignment reports.
"""

from __future__ import annotations

from scripts.compare_protocol_to_baselines import main


if __name__ == "__main__":
    raise SystemExit(main())
