# Human Playtest, OOD, And QD Archive Protocol

Last updated: 2026-06-02

This protocol closes executable tooling gaps. It does not claim that the
human study, trained OOD probe, or final QD run has already been executed.

## 1. Human Playtest Telemetry

Obtain the required ethics or institutional approval before recruiting
participants. Record informed consent and use pseudonymous participant IDs.
Do not store names, emails, or other direct identifiers in repository
telemetry.

Start each human session through
`PlaytestTelemetryCollector.start_human_session(...)`:

```python
from src.utils.playtest_telemetry import PlaytestTelemetryCollector

collector = PlaytestTelemetryCollector("results/playtest/human")
collector.start_human_session(
    "session_001",
    participant_id="P001",
    study_id="hmolqd_playtest_v1",
    consent_recorded=True,
    context={"persona": "observed", "level_id": "generated_001"},
)
```

Pass the collector into the replay engine used by the playtest. After the
sessions are complete, validate provenance before calibration:

```bash
python scripts/validate_human_playtest_telemetry.py \
  --telemetry results/playtest/human \
  --output results/playtest/human_playtest_manifest.json

python scripts/calibrate_pcbs_personas_from_telemetry.py \
  --telemetry results/playtest/human \
  --pcbs-sweep-csv results/pcbs_persona_map_sweep/pcbs_persona_map_sweep.csv \
  --output-dir results/pcbs_telemetry_calibration
```

Synthetic, replay-only, and simulated P-CBS traces may be useful for software
tests, but they are not human evidence and must not be reported as playtest
validation.

## 2. Synthetic Structural OOD Probe

Run the schema smoke probe first:

```bash
python scripts/run_synthetic_metroidvania_ood_probe.py \
  --output-dir results/synthetic_metroidvania_ood_schema_smoke
```

The smoke output is explicitly labeled `schema_smoke_random_weights`. It is
not publishable OOD evidence.

Run the checkpoint-backed condition-encoder probe:

```bash
python scripts/run_synthetic_metroidvania_ood_probe.py \
  --condition-encoder-checkpoint outputs/zelda_hmolqd/checkpoints/condition_encoder.pth \
  --output-dir results/synthetic_metroidvania_ood_encoder
```

For full room-generation evidence, supply the trained room stack:

```bash
python scripts/run_synthetic_metroidvania_ood_probe.py \
  --condition-encoder-checkpoint outputs/zelda_hmolqd/checkpoints/condition_encoder.pth \
  --vqvae-checkpoint outputs/zelda_hmolqd/checkpoints/vqvae/vqvae_pretrained.pth \
  --diffusion-checkpoint outputs/zelda_hmolqd/checkpoints/diffusion/best_model.pth \
  --generate-rooms \
  --output-dir results/synthetic_metroidvania_ood_rooms
```

Report the structural-shift descriptor delta, finite conditioning outputs,
embedding shift, generated-room count, and downstream oracle playability.
The synthetic topology supplements the Zelda dungeon-9 holdout; it does not
replace an external-domain dataset.

## 3. QD Archive Diversity Heatmaps

For a JSON archive export:

```bash
python scripts/visualize_qd_archive.py \
  --archive results/map_elites/archive.json \
  --output-dir results/qd_archive_analysis
```

Create the portable analysis snapshot with
`evaluator.export_archive_json("results/map_elites/archive.json")` after the
publication run. The JSON snapshot omits full dungeon solution objects and is
intended for reporting, not warm starts.

The runtime MAP-Elites evaluator currently persists trusted local pickle
archives. Analyze those only when the archive was produced locally:

```bash
python scripts/visualize_qd_archive.py \
  --archive results/map_elites/runtime_map_elites.pkl \
  --trust-pickle \
  --output-dir results/qd_archive_analysis
```

The analyzer emits `qd_archive_diversity_report.json` and
`qd_archive_pairwise_heatmaps.png`. Four-dimensional archives are rendered as
all pairwise 2D projections.

## Research Anchors

- MAP-Elites behavioral dimensions: <https://arxiv.org/abs/1504.04909>
- WILDS distribution-shift evaluation principle:
  <https://arxiv.org/abs/2012.07421>
- HHS informed-consent FAQ:
  <https://www.hhs.gov/ohrp/regulations-and-policy/guidance/faq/informed-consent/index.html>
