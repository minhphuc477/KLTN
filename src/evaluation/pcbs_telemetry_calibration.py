"""Telemetry-driven calibration helpers for P-CBS personas."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.simulation.cognitive_bounded_search import AgentPersona, PersonaConfig

PCBS_CALIBRATION_PROVENANCE = {
    "hard_oracle": "full_state_astar",
    "bounded_agent": "p_cbs",
    "diagnostic_solvers_excluded": ["bidirectional_astar", "forward_lpa_replanning", "greedy"],
    "note": (
        "Persona calibration must use hard-oracle A* and P-CBS outputs from "
        "the same semantic-grid validator. Bidirectional and replanning "
        "diagnostics are not calibration anchors."
    ),
}


NUMERIC_KEYS = (
    "success_rate",
    "avg_steps",
    "avg_unique_tiles",
    "avg_revisit_rate",
    "avg_confusion_index",
    "avg_navigation_entropy",
    "avg_cognitive_load",
    "avg_path_efficiency",
    "avg_decision_time_ms",
)


@dataclass
class TelemetrySessionMetrics:
    """One normalized human or simulated navigation session."""

    persona: str
    source: str
    success: Optional[bool] = None
    steps: Optional[float] = None
    unique_tiles: Optional[float] = None
    revisit_rate: Optional[float] = None
    confusion_index: Optional[float] = None
    navigation_entropy: Optional[float] = None
    cognitive_load: Optional[float] = None
    path_efficiency: Optional[float] = None
    decision_time_ms: Optional[float] = None


@dataclass
class TelemetryAggregate:
    """Aggregate target metrics used to tune a persona."""

    persona: str
    sessions: int
    success_rate: Optional[float] = None
    avg_steps: Optional[float] = None
    avg_unique_tiles: Optional[float] = None
    avg_revisit_rate: Optional[float] = None
    avg_confusion_index: Optional[float] = None
    avg_navigation_entropy: Optional[float] = None
    avg_cognitive_load: Optional[float] = None
    avg_path_efficiency: Optional[float] = None
    avg_decision_time_ms: Optional[float] = None


def load_telemetry_sessions(paths: Sequence[str | Path]) -> List[TelemetrySessionMetrics]:
    """Load playtest telemetry from JSON, JSONL, or CSV files/directories."""
    sessions: List[TelemetrySessionMetrics] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            child_paths = sorted(
                p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".csv"}
            )
            sessions.extend(load_telemetry_sessions(child_paths))
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            sessions.extend(_load_jsonl(path))
        elif suffix == ".json":
            sessions.extend(_load_json(path))
        elif suffix == ".csv":
            sessions.extend(_load_csv(path))
    return sessions


def aggregate_sessions(sessions: Iterable[TelemetrySessionMetrics]) -> Dict[str, TelemetryAggregate]:
    """Aggregate normalized session metrics by persona label."""
    grouped: Dict[str, List[TelemetrySessionMetrics]] = {}
    for session in sessions:
        grouped.setdefault(_normalize_persona(session.persona), []).append(session)

    aggregates: Dict[str, TelemetryAggregate] = {}
    for persona, rows in grouped.items():
        successes = [1.0 if row.success else 0.0 for row in rows if row.success is not None]
        aggregates[persona] = TelemetryAggregate(
            persona=persona,
            sessions=len(rows),
            success_rate=_safe_mean(successes),
            avg_steps=_safe_mean(row.steps for row in rows),
            avg_unique_tiles=_safe_mean(row.unique_tiles for row in rows),
            avg_revisit_rate=_safe_mean(row.revisit_rate for row in rows),
            avg_confusion_index=_safe_mean(row.confusion_index for row in rows),
            avg_navigation_entropy=_safe_mean(row.navigation_entropy for row in rows),
            avg_cognitive_load=_safe_mean(row.cognitive_load for row in rows),
            avg_path_efficiency=_safe_mean(row.path_efficiency for row in rows),
            avg_decision_time_ms=_safe_mean(row.decision_time_ms for row in rows),
        )
    return aggregates


def calibrate_persona_overrides(
    telemetry_targets: Mapping[str, TelemetryAggregate],
    *,
    pcbs_baselines: Optional[Mapping[str, TelemetryAggregate]] = None,
    personas: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Recommend PersonaConfig overrides from human telemetry.

    When a P-CBS sweep baseline is provided, adjustments are based on the
    target-minus-baseline gap. Without a baseline, adjustments use conservative
    anchors derived from the built-in persona.
    """
    selected = [_normalize_persona(p) for p in personas] if personas else _default_persona_names()
    global_target = telemetry_targets.get("observed")
    results: Dict[str, Dict[str, Any]] = {}
    for persona in selected:
        target = telemetry_targets.get(persona) or global_target
        if target is None:
            continue
        baseline = (pcbs_baselines or {}).get(persona)
        base_config = _base_config(persona)
        calibrated = _calibrate_single(base_config, target, baseline)
        base_payload = asdict(base_config)
        calibrated_payload = asdict(calibrated)
        overrides = {
            key: value
            for key, value in calibrated_payload.items()
            if base_payload.get(key) != value
        }
        results[persona] = {
            "base_persona": persona,
            "target_sessions": int(target.sessions),
            "target_metrics": asdict(target),
            "baseline_metrics": asdict(baseline) if baseline is not None else None,
            "calibration_provenance": dict(PCBS_CALIBRATION_PROVENANCE),
            "overrides": overrides,
            "calibrated_config": calibrated_payload,
        }
    return results


def render_calibration_markdown(
    targets: Mapping[str, TelemetryAggregate],
    overrides: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render a compact calibration report for papers and audits."""
    lines = [
        "# P-CBS Telemetry Calibration",
        "",
        "This report derives persona overrides from local playtest telemetry.",
        "Use it as an empirical calibration artifact, not as a user-study substitute.",
        "",
        "| Persona | Sessions | Success % | Revisit rate | Confusion | Path efficiency | Key overrides |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for persona in sorted(overrides):
        payload = overrides[persona]
        target = targets.get(persona) or targets.get("observed")
        if target is None:
            continue
        changed = payload.get("overrides", {})
        keys = ", ".join(sorted(changed.keys())[:8])
        if len(changed) > 8:
            keys += ", ..."
        lines.append(
            "| {persona} | {sessions} | {success} | {revisit} | {confusion} | {efficiency} | {keys} |".format(
                persona=persona,
                sessions=int(target.sessions),
                success=_fmt_pct(target.success_rate),
                revisit=_fmt_float(target.avg_revisit_rate),
                confusion=_fmt_float(target.avg_confusion_index),
                efficiency=_fmt_float(target.avg_path_efficiency),
                keys=keys or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "The calibrator normalizes telemetry into success, path effort, revisits, confusion, entropy, cognitive load, and decision-time targets. "
            "If a P-CBS sweep CSV is supplied, it adjusts each built-in persona toward the human target relative to that simulated baseline. "
            "Without a sweep baseline, it uses conservative built-in anchors so the output remains stable.",
            "",
            "Calibration provenance: hard oracle = full-state A*; bounded agent = P-CBS; bidirectional and replanning diagnostics are excluded from persona anchoring.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> List[TelemetrySessionMetrics]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_session_from_json(item, str(path)) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        return [
            _session_from_json(item, str(path))
            for item in payload["sessions"]
            if isinstance(item, dict)
        ]
    if isinstance(payload, dict):
        return [_session_from_json(payload, str(path))]
    return []


def _load_jsonl(path: Path) -> List[TelemetrySessionMetrics]:
    rows: List[TelemetrySessionMetrics] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                rows.append(_session_from_json(payload, str(path)))
    return rows


def _load_csv(path: Path) -> List[TelemetrySessionMetrics]:
    rows: List[TelemetrySessionMetrics] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(_session_from_row(row, str(path)))
    return rows


def _session_from_json(payload: Mapping[str, Any], source: str) -> TelemetrySessionMetrics:
    context = _dict(payload.get("context"))
    summary = _dict(payload.get("summary"))
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    persona = _first_text(
        summary.get("persona"),
        context.get("persona"),
        context.get("player_persona"),
        payload.get("persona"),
        "observed",
    )
    positions = [_position_from_event(evt) for evt in events if isinstance(evt, Mapping)]
    positions = [pos for pos in positions if pos is not None]
    steps = _first_number(
        summary.get("total_steps"),
        summary.get("steps"),
        summary.get("path_length"),
        summary.get("trajectory_length"),
        len(positions) if positions else None,
    )
    unique_tiles = _first_number(
        summary.get("unique_tiles"),
        summary.get("unique_tiles_visited"),
        len(set(positions)) if positions else None,
    )
    revisit_rate = _first_number(summary.get("revisit_rate"))
    if revisit_rate is None and steps and unique_tiles is not None and steps > 0:
        revisit_rate = max(0.0, (float(steps) - float(unique_tiles)) / float(steps))
    duration_sec = _first_number(summary.get("duration_sec"), payload.get("duration_sec"))
    if duration_sec is None:
        times = [_first_number(_dict(evt).get("t_rel_sec")) for evt in events if isinstance(evt, Mapping)]
        finite_times = [t for t in times if t is not None]
        duration_sec = max(finite_times) if finite_times else None
    decision_time_ms = _first_number(summary.get("decision_time_ms"), summary.get("avg_decision_time_ms"))
    if decision_time_ms is None and duration_sec is not None and steps and steps > 0:
        decision_time_ms = (float(duration_sec) * 1000.0) / float(steps)
    return TelemetrySessionMetrics(
        persona=persona,
        source=source,
        success=_first_bool(summary.get("success"), payload.get("success"), payload.get("status")),
        steps=steps,
        unique_tiles=unique_tiles,
        revisit_rate=revisit_rate,
        confusion_index=_first_number(summary.get("confusion_index")),
        navigation_entropy=_first_number(summary.get("navigation_entropy")),
        cognitive_load=_first_number(summary.get("cognitive_load")),
        path_efficiency=_path_efficiency(summary),
        decision_time_ms=decision_time_ms,
    )


def _session_from_row(row: Mapping[str, Any], source: str) -> TelemetrySessionMetrics:
    persona = _first_text(row.get("persona"), row.get("player_persona"), "observed")
    steps = _first_number(
        row.get("total_steps"),
        row.get("steps"),
        row.get("pcbs_trajectory_length"),
        row.get("trajectory_length"),
        row.get("path_length"),
        row.get("pcbs_path_length"),
    )
    unique_tiles = _first_number(row.get("unique_tiles"), row.get("unique_tiles_visited"))
    revisit_rate = _first_number(row.get("revisit_rate"))
    if revisit_rate is None and steps and unique_tiles is not None and steps > 0:
        revisit_rate = max(0.0, (float(steps) - float(unique_tiles)) / float(steps))
    time_ms = _first_number(row.get("time_ms"), row.get("duration_ms"))
    decision_time_ms = _first_number(row.get("decision_time_ms"), row.get("avg_decision_time_ms"))
    if decision_time_ms is None and time_ms is not None and steps and steps > 0:
        decision_time_ms = float(time_ms) / float(steps)
    return TelemetrySessionMetrics(
        persona=persona,
        source=source,
        success=_first_bool(row.get("success"), row.get("pcbs_success"), row.get("status"), row.get("pcbs_status")),
        steps=steps,
        unique_tiles=unique_tiles,
        revisit_rate=revisit_rate,
        confusion_index=_first_number(row.get("confusion_index")),
        navigation_entropy=_first_number(row.get("navigation_entropy")),
        cognitive_load=_first_number(row.get("cognitive_load")),
        path_efficiency=_path_efficiency(row),
        decision_time_ms=decision_time_ms,
    )


def _calibrate_single(
    base: PersonaConfig,
    target: TelemetryAggregate,
    baseline: Optional[TelemetryAggregate],
) -> PersonaConfig:
    data = asdict(base)
    anchor = baseline or _anchor_from_config(base)

    confusion_gap = _gap(target.avg_confusion_index, anchor.avg_confusion_index, scale=3.0)
    revisit_gap = _gap(target.avg_revisit_rate, anchor.avg_revisit_rate, scale=1.0)
    efficiency_gap = _gap(target.avg_path_efficiency, anchor.avg_path_efficiency, scale=1.0)
    load_gap = _gap(target.avg_cognitive_load, anchor.avg_cognitive_load, scale=2.5)
    entropy_gap = _gap(target.avg_navigation_entropy, anchor.avg_navigation_entropy, scale=2.0)
    success_gap = _gap(target.success_rate, anchor.success_rate, scale=1.0)
    decision_gap = _gap(target.avg_decision_time_ms, anchor.avg_decision_time_ms, scale=1000.0)

    bounded_gap = _clamp(
        0.35 * confusion_gap
        + 0.25 * revisit_gap
        - 0.20 * efficiency_gap
        + 0.10 * load_gap
        + 0.10 * entropy_gap
        - 0.15 * success_gap,
        -1.0,
        1.0,
    )

    data["memory_capacity"] = int(_clamp(round(base.memory_capacity - 2.0 * bounded_gap), 3, 12))
    data["memory_decay_rate"] = _clamp(base.memory_decay_rate - 0.08 * bounded_gap, 0.70, 1.0)
    data["vision_radius"] = int(_clamp(round(base.vision_radius - 1.5 * bounded_gap), 3, 12))
    data["vision_accuracy"] = _clamp(base.vision_accuracy - 0.06 * bounded_gap, 0.75, 0.99)
    data["satisficing_threshold"] = _clamp(base.satisficing_threshold - 0.08 * bounded_gap, 0.45, 1.0)
    data["random_tiebreaker"] = _clamp(base.random_tiebreaker + 0.16 * bounded_gap + 0.05 * entropy_gap, 0.0, 0.45)
    data["goal_weight"] = _clamp(base.goal_weight - 0.12 * bounded_gap, 0.05, 1.0)
    data["curiosity_weight"] = _clamp(base.curiosity_weight + 0.10 * entropy_gap + 0.06 * revisit_gap, 0.0, 1.0)
    data["risk_weight"] = _clamp(base.risk_weight + 0.10 * load_gap, 0.0, 1.0)
    data["revisit_penalty_weight"] = _clamp(base.revisit_penalty_weight - 0.12 * revisit_gap, 0.0, 0.8)
    data["puzzle_complexity_weight"] = _clamp(base.puzzle_complexity_weight + 0.10 * load_gap, 0.0, 1.0)
    data["conditional_uncertainty_penalty_weight"] = _clamp(
        base.conditional_uncertainty_penalty_weight + 0.10 * load_gap,
        0.0,
        1.0,
    )
    data["deliberation_budget"] = _clamp(base.deliberation_budget + 1.5 * decision_gap + 0.8 * load_gap, 3.0, 14.0)
    data["deliberation_trigger"] = _clamp(base.deliberation_trigger - 0.10 * bounded_gap, 0.20, 0.95)
    data["frustration_sensitivity"] = _clamp(base.frustration_sensitivity + 0.15 * load_gap, 0.0, 0.8)

    heuristic = dict(base.heuristic_weights)
    heuristic["goal_seeking"] = _clamp(float(heuristic.get("goal_seeking", 1.0)) - 0.20 * bounded_gap, 0.0, 2.5)
    heuristic["curiosity"] = _clamp(float(heuristic.get("curiosity", 0.5)) + 0.20 * entropy_gap, 0.0, 2.5)
    heuristic["recency"] = _clamp(float(heuristic.get("recency", 0.5)) + 0.20 * revisit_gap, 0.0, 2.5)
    heuristic["safety"] = _clamp(float(heuristic.get("safety", 0.5)) + 0.20 * load_gap, 0.0, 2.5)
    data["heuristic_weights"] = heuristic

    return PersonaConfig(**data)


def _anchor_from_config(config: PersonaConfig) -> TelemetryAggregate:
    bounded = _clamp(
        (7 - config.memory_capacity) / 8.0
        + (0.95 - config.memory_decay_rate)
        + config.random_tiebreaker
        + max(0.0, 5 - config.vision_radius) / 8.0,
        0.0,
        1.0,
    )
    return TelemetryAggregate(
        persona=_normalize_persona(config.name),
        sessions=0,
        success_rate=_clamp(0.92 - 0.35 * bounded, 0.35, 0.98),
        avg_revisit_rate=_clamp(0.08 + 0.35 * bounded, 0.0, 0.9),
        avg_confusion_index=_clamp(0.25 + 1.8 * bounded, 0.0, 3.5),
        avg_navigation_entropy=_clamp(0.45 + 1.0 * config.random_tiebreaker, 0.0, 2.0),
        avg_cognitive_load=_clamp(0.45 + 1.4 * bounded, 0.0, 2.5),
        avg_path_efficiency=_clamp(1.0 - 0.65 * bounded, 0.0, 1.0),
        avg_decision_time_ms=_clamp(300.0 + 80.0 * config.deliberation_budget, 100.0, 2000.0),
    )


def _path_efficiency(row: Mapping[str, Any]) -> Optional[float]:
    direct = _first_number(
        row.get("path_efficiency"),
        row.get("path_efficiency_ratio"),
        row.get("avg_path_efficiency"),
    )
    if direct is not None:
        return _clamp(direct, 0.0, 1.0)
    path_length = _first_number(row.get("path_length"), row.get("pcbs_path_length"), row.get("trajectory_length"))
    oracle_length = _first_number(row.get("oracle_path_length"), row.get("optimal_path_length"))
    if path_length is not None and oracle_length and oracle_length > 0:
        if path_length <= 0:
            return 0.0
        return _clamp(float(oracle_length) / float(path_length), 0.0, 1.0)
    return None


def _position_from_event(event: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    raw = event.get("position")
    if raw is None and isinstance(event.get("payload"), Mapping):
        raw = event["payload"].get("position")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None


def _base_config(persona: str) -> PersonaConfig:
    normalized = _normalize_persona(persona)
    for candidate in AgentPersona:
        if candidate.value == normalized:
            return PersonaConfig.get_persona(candidate)
    return PersonaConfig.get_persona(AgentPersona.BALANCED)


def _default_persona_names() -> List[str]:
    return [persona.value for persona in AgentPersona]


def _normalize_persona(value: Any) -> str:
    raw = str(value or "observed").strip().lower()
    aliases = {
        "balanced": "balanced",
        "human": "observed",
        "player": "observed",
        "playtester": "observed",
        "greedy (static)": "greedy",
        "speedrunner": "speedrunner",
        "speed": "speedrunner",
    }
    return aliases.get(raw, raw)


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    cleaned: List[float] = []
    for value in values:
        numeric = _first_number(value)
        if numeric is not None and math.isfinite(numeric):
            cleaned.append(float(numeric))
    return float(fmean(cleaned)) if cleaned else None


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "observed"


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def _first_bool(*values: Any) -> Optional[bool]:
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "success", "succeeded", "solved", "completed"}:
            return True
        if text in {"0", "false", "no", "failed", "timeout", "interrupted"}:
            return False
    return None


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _gap(target: Optional[float], baseline: Optional[float], *, scale: float) -> float:
    if target is None or baseline is None or scale <= 0:
        return 0.0
    return _clamp((float(target) - float(baseline)) / float(scale), -1.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _fmt_float(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.1f}"


__all__ = [
    "TelemetrySessionMetrics",
    "TelemetryAggregate",
    "PCBS_CALIBRATION_PROVENANCE",
    "load_telemetry_sessions",
    "aggregate_sessions",
    "calibrate_persona_overrides",
    "render_calibration_markdown",
]
