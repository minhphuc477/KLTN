"""Pydantic v2 validation models for H-MOLQD configuration sections."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.config_system import CURRENT_CONFIG_VERSION, CONFIG_FIELDS, validate_config
from src.pipeline.config import GraphConfig, ModelConfig, PipelineConfig, SamplerConfig

TopologyConfig = GraphConfig


def _annotation_for_field(field_type: type, allow_none: bool) -> Any:
    annotation: Any = field_type
    if allow_none:
        annotation = Optional[annotation]
    return annotation


def _field_kwargs(config_field: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"description": str(config_field.help)}
    if config_field.field_type in {int, float} and config_field.min_value is not None:
        kwargs["ge"] = config_field.min_value
    if config_field.field_type in {int, float} and config_field.max_value is not None:
        kwargs["le"] = config_field.max_value
    return kwargs


def _build_section_model(section: str) -> Type[BaseModel]:
    fields: Dict[str, Tuple[Any, Any]] = {}
    prefix = f"{section}."
    for config_field in CONFIG_FIELDS:
        if not config_field.path.startswith(prefix):
            continue
        name = config_field.path[len(prefix):]
        if "." in name:
            continue
        annotation = _annotation_for_field(config_field.field_type, bool(config_field.allow_none))
        fields[name] = (
            annotation,
            Field(default=config_field.default, **_field_kwargs(config_field)),
        )
    return create_model(
        f"{section.title().replace('_', '')}Schema",
        __config__=ConfigDict(extra="forbid", validate_assignment=True),
        **fields,
    )


TrainingSchema = _build_section_model("training")
RuntimeSchema = _build_section_model("runtime")
DatasetSchema = _build_section_model("dataset")
VQVaeSchema = _build_section_model("vqvae")
DiffusionSchema = _build_section_model("diffusion")
GenerationSchema = _build_section_model("generation")
FastSamplerSchema = _build_section_model("fast_sampler")
MaskedRoomSchema = _build_section_model("masked_room")
TopologySchema = _build_section_model("topology")
DistributedSchema = _build_section_model("distributed")


class HMOLQDConfigSchema(BaseModel):
    """Full experiment config schema with strict section-level extra checks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_version: int = Field(default=CURRENT_CONFIG_VERSION, ge=1)
    training: TrainingSchema = Field(default_factory=TrainingSchema)
    runtime: RuntimeSchema = Field(default_factory=RuntimeSchema)
    dataset: DatasetSchema = Field(default_factory=DatasetSchema)
    vqvae: VQVaeSchema = Field(default_factory=VQVaeSchema)
    diffusion: DiffusionSchema = Field(default_factory=DiffusionSchema)
    generation: GenerationSchema = Field(default_factory=GenerationSchema)
    fast_sampler: FastSamplerSchema = Field(default_factory=FastSamplerSchema)
    masked_room: MaskedRoomSchema = Field(default_factory=MaskedRoomSchema)
    topology: TopologySchema = Field(default_factory=TopologySchema)
    distributed: DistributedSchema = Field(default_factory=DistributedSchema)

    @model_validator(mode="after")
    def _cross_validate_with_runtime_schema(self) -> "HMOLQDConfigSchema":
        validate_config(self.model_dump())
        return self


def validate_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a config payload with Pydantic and existing cross-field checks."""
    model = HMOLQDConfigSchema.model_validate(payload)
    return validate_config(model.model_dump())


__all__ = [
    "PipelineConfig",
    "ModelConfig",
    "SamplerConfig",
    "GraphConfig",
    "TopologyConfig",
    "TrainingSchema",
    "RuntimeSchema",
    "DatasetSchema",
    "VQVaeSchema",
    "DiffusionSchema",
    "GenerationSchema",
    "FastSamplerSchema",
    "MaskedRoomSchema",
    "TopologySchema",
    "DistributedSchema",
    "HMOLQDConfigSchema",
    "validate_config_payload",
]
