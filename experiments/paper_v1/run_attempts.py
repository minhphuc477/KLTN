"""Execute an externally sealed sampling plan into a fresh attempt ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.publication.attempt_runner import AttemptLedger, AttemptSpec, execute_attempts
from src.publication.pipeline_attempt import CanonicalAttemptPipeline
from src.publication.protocol import payload_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True,
                        help="Previously sealed canonical JSON digest, not inferred from current contents.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    if payload_sha256(payload) != args.plan_sha256:
        raise ValueError("Sampling plan does not match its previously sealed digest.")
    if payload.get("schema_version") != "2.0":
        raise ValueError("Unsupported sampling-plan schema.")
    specs = [AttemptSpec(**entry) for entry in payload["attempts"]]
    from src.publication.execution_provenance import verify_execution_provenance
    verify_execution_provenance(payload.get("execution_provenance"))
    ledger = AttemptLedger.create(args.output_root, specs, execution_provenance=payload["execution_provenance"])
    pipeline = CanonicalAttemptPipeline()
    from src.publication.control_attempt import ControlAttemptPipeline
    from src.publication.protocol import CONTROL_MODELS
    controls = ControlAttemptPipeline()
    def adapter(spec):
        return controls if spec.metadata["model_id"] in CONTROL_MODELS else pipeline
    execute_attempts(ledger, generate=lambda spec: adapter(spec).generate(spec),
                     validator_factory=lambda spec, generated: adapter(spec).validator_factory(spec, generated))
    print(f"Completed all {len(specs)} sealed attempt records under {ledger.root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
