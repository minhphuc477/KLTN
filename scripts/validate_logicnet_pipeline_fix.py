#!/usr/bin/env python
"""Quick end-to-end validation of LogicNet fixes in the ablation pipeline."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_quick_ablation():
    """Run an n=2 paired-seed ablation to verify LogicNet can execute."""
    from scripts.run_ablation_study import AblationStudy, ExperimentConfig

    print("=" * 70)
    print("Running n=2 quick ablation with fixed LogicNet")
    print("=" * 70)

    study = AblationStudy(
        name="logicnet_fix_validation_n2",
        output_dir="results/logicnet_fix_validation",
        paired_seeds=[
            (42, 42),
            (123, 123),
        ],
    )

    configs = {
        "FULL": ExperimentConfig(
            name="FULL_with_logic",
            logic_guidance_scale=1.0,
        ),
        "NO_LOGIC": ExperimentConfig(
            name="NO_LOGIC",
            logic_guidance_scale=0.0,
        ),
    }

    results = study.run(configs)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    for metric, values in results.items():
        if not (isinstance(values, dict) and "FULL" in values and "NO_LOGIC" in values):
            continue

        full_mean = values["FULL"]["mean"]
        full_std = values["FULL"]["std"]
        logic_mean = values["NO_LOGIC"]["mean"]
        logic_std = values["NO_LOGIC"]["std"]
        diff = full_mean - logic_mean

        print(f"\n{metric}:")
        print(f"  FULL (logic=1.0):  {full_mean:.4f} +/- {full_std:.4f}")
        print(f"  NO_LOGIC (logic=0): {logic_mean:.4f} +/- {logic_std:.4f}")
        print(f"  Difference:        {diff:+.4f}")

        if full_std > 0 and logic_std > 0:
            pooled_std = ((full_std**2 + logic_std**2) / 2) ** 0.5
            t_stat = diff / (pooled_std + 1e-8)
            print(f"  t-stat (approx):   {t_stat:.2f}")

    return results


if __name__ == "__main__":
    try:
        run_quick_ablation()
        print("\n[OK] Quick validation test completed successfully")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FAIL] ERROR: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
