"""Publication experiment contracts.

The package is deliberately independent from training and generation entry
points.  It defines the frozen study design and machine-readable result
contracts without importing heavyweight model modules.
"""

from .protocol import (
    PUBLICATION_VARIANTS,
    ModelVariant,
    OuterFold,
    apply_publication_variant,
    build_leave_one_dungeon_out_folds,
    build_protocol_manifest,
    seal_protocol_manifest,
)
from .records import (
    PublicationResultRecord,
    StageResult,
    changed_tile_metrics,
)
from .analysis import (
    aggregate_publication_records,
    holm_adjust_pvalues,
    paired_exact_validity_by_fold,
    paired_guidance_validity_by_fold,
)
from .dataset_provenance import verify_dataset_manifest


def __getattr__(name: str):
    """Load the pipeline-dependent stage bridge only when explicitly requested."""
    if name in {"build_publication_result_record", "extract_publication_stage_grids"}:
        from .stage_bridge import (
            build_publication_result_record,
            extract_publication_stage_grids,
        )

        return {
            "build_publication_result_record": build_publication_result_record,
            "extract_publication_stage_grids": extract_publication_stage_grids,
        }[name]
    raise AttributeError(name)

__all__ = [
    "PUBLICATION_VARIANTS",
    "ModelVariant",
    "OuterFold",
    "PublicationResultRecord",
    "StageResult",
    "apply_publication_variant",
    "aggregate_publication_records",
    "holm_adjust_pvalues",
    "build_publication_result_record",
    "build_leave_one_dungeon_out_folds",
    "build_protocol_manifest",
    "seal_protocol_manifest",
    "changed_tile_metrics",
    "extract_publication_stage_grids",
    "paired_exact_validity_by_fold",
    "paired_guidance_validity_by_fold",
    "verify_dataset_manifest",
]
