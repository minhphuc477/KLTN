"""Check the primary endpoint against source training labels before training."""

import argparse
import json
from pathlib import Path

from src.publication.data_support import preflight_training_manifest, verify_training_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--execution-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    if args.execution_only:
        verify_training_inputs(manifest)
    else:
        preflight_training_manifest(manifest)
    print("Execution inputs verified." if args.execution_only else "Primary-endpoint training-target support check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
