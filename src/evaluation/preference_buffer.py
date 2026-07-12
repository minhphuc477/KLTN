"""Bounded same-condition preference replay for QD-to-DPO experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import networkx as nx
import numpy as np
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
from src.utils.checkpoint import atomic_torch_save


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        enum_value = value.value
        if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
            return enum_value
        return value.name
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_value(item) for item in value), key=repr)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def serialize_condition_graph(graph: nx.Graph) -> Dict[str, Any]:
    """Serialize graph semantics deterministically without executable objects."""
    nodes = [
        {
            "id_type": type(node_id).__name__,
            "id": repr(node_id),
            "attrs": _canonical_value(dict(attrs)),
        }
        for node_id, attrs in sorted(graph.nodes(data=True), key=lambda item: (type(item[0]).__name__, repr(item[0])))
    ]
    edges = [
        {
            "source_type": type(source).__name__,
            "source": repr(source),
            "target_type": type(target).__name__,
            "target": repr(target),
            "attrs": _canonical_value(dict(attrs)),
        }
        for source, target, attrs in sorted(
            graph.edges(data=True),
            key=lambda item: (
                type(item[0]).__name__,
                repr(item[0]),
                type(item[1]).__name__,
                repr(item[1]),
            ),
        )
    ]
    return {
        "directed": bool(graph.is_directed()),
        "graph_attrs": _canonical_value(dict(graph.graph)),
        "nodes": nodes,
        "edges": edges,
    }


def condition_graph_fingerprint(graph: nx.Graph) -> str:
    payload = serialize_condition_graph(graph)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deserialize_condition_graph(payload: Mapping[str, Any]) -> nx.Graph:
    """Rebuild a NetworkX graph from :func:`serialize_condition_graph` output."""
    graph: nx.Graph = nx.DiGraph() if bool(payload.get("directed", True)) else nx.Graph()
    graph.graph.update(dict(payload.get("graph_attrs", {}) or {}))

    def _parse_id(value: Any) -> Any:
        text = str(value)
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

    for node in payload.get("nodes", []) or []:
        if not isinstance(node, Mapping):
            continue
        graph.add_node(_parse_id(node.get("id")), **dict(node.get("attrs", {}) or {}))
    for edge in payload.get("edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        graph.add_edge(
            _parse_id(edge.get("source")),
            _parse_id(edge.get("target")),
            **dict(edge.get("attrs", {}) or {}),
        )
    return graph


@dataclass
class PreferenceCandidate:
    condition_id: str
    room_id: str
    tiles: np.ndarray
    score: float
    solvable: bool
    graph_payload: Dict[str, Any]
    metadata: Dict[str, Any]


class QDPreferenceBuffer:
    """Uniform bounded replay that creates only same-condition comparisons."""

    def __init__(self, max_candidates: int, *, seed: Optional[int] = None) -> None:
        self.max_candidates = int(max(0, max_candidates))
        self.rng = np.random.default_rng(seed)
        self.candidates: list[PreferenceCandidate] = []
        self.total_seen = 0

    @property
    def enabled(self) -> bool:
        return self.max_candidates > 0

    def add_rooms(
        self,
        rooms: Mapping[Any, Any],
        *,
        mission_graph: nx.Graph,
        score: float,
        solvable: bool,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.enabled or not np.isfinite(float(score)):
            return
        condition_id = condition_graph_fingerprint(mission_graph)
        graph_payload = serialize_condition_graph(mission_graph)
        for room_id, room in rooms.items():
            raw_grid = getattr(room, "raw_neural_grid", None)
            if raw_grid is None:
                # DPO is intended to improve the neural generator, not imitate
                # symbolic repair. Final/cleaned grids are therefore not valid
                # substitutes for an absent pre-repair sample.
                continue
            tiles = np.asarray(raw_grid, dtype=np.int64)
            if tuple(tiles.shape) != (int(ROOM_HEIGHT), int(ROOM_WIDTH)):
                continue
            room_metrics = getattr(room, "metrics", {})
            if not isinstance(room_metrics, Mapping):
                room_metrics = {}
            changed_tiles = float(room_metrics.get("raw_neural_to_final_tiles_changed", 0.0) or 0.0)
            repair_burden = float(np.clip(changed_tiles / max(1, tiles.size), 0.0, 1.0))
            candidate_score = float(score) - repair_burden
            candidate_metadata = {
                str(key): _canonical_value(value)
                for key, value in dict(metadata or {}).items()
            }
            candidate_metadata.update(
                {
                    "global_qd_score": float(score),
                    "room_repair_burden": repair_burden,
                }
            )
            candidate = PreferenceCandidate(
                condition_id=condition_id,
                room_id=repr(room_id),
                tiles=tiles.copy(),
                score=candidate_score,
                solvable=bool(solvable),
                graph_payload=graph_payload,
                metadata=candidate_metadata,
            )
            self._reservoir_add(candidate)

    def _reservoir_add(self, candidate: PreferenceCandidate) -> None:
        self.total_seen += 1
        if len(self.candidates) < self.max_candidates:
            self.candidates.append(candidate)
            return
        replacement = int(self.rng.integers(0, self.total_seen))
        if replacement < self.max_candidates:
            self.candidates[replacement] = candidate

    def build_pairs(self, *, min_score_margin: float = 0.05) -> list[Dict[str, Any]]:
        grouped: Dict[tuple[str, str], list[PreferenceCandidate]] = {}
        for candidate in self.candidates:
            grouped.setdefault((candidate.condition_id, candidate.room_id), []).append(candidate)

        pairs: list[Dict[str, Any]] = []
        for (condition_id, room_id), candidates in sorted(grouped.items()):
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates, key=lambda item: (bool(item.solvable), float(item.score)))
            rejected = ordered[0]
            chosen = ordered[-1]
            margin = float(chosen.score) - float(rejected.score)
            if chosen.solvable and not rejected.solvable:
                margin = max(margin, 1.0)
            if margin < float(min_score_margin) or np.array_equal(chosen.tiles, rejected.tiles):
                continue
            pairs.append(
                {
                    "condition_id": condition_id,
                    "room_id": room_id,
                    "preferred_tiles": chosen.tiles,
                    "rejected_tiles": rejected.tiles,
                    "preferred_score": float(chosen.score),
                    "rejected_score": float(rejected.score),
                    "preferred_solvable": bool(chosen.solvable),
                    "rejected_solvable": bool(rejected.solvable),
                    "graph_payload": chosen.graph_payload,
                    "preferred_metadata": chosen.metadata,
                    "rejected_metadata": rejected.metadata,
                }
            )
        return pairs

    def export_raw_pairs(self, path: str | Path, *, min_score_margin: float = 0.05) -> Path:
        pairs = self.build_pairs(min_score_margin=min_score_margin)
        if not pairs:
            raise ValueError(
                "No valid same-condition preference pairs are available. Generate at least two "
                "different room samples for the same mission graph and room ID."
            )
        payload = {
            "format": "hmolqd_raw_room_preferences_v1",
            "preferred_tiles": torch.from_numpy(np.stack([pair["preferred_tiles"] for pair in pairs])).long().unsqueeze(1),
            "rejected_tiles": torch.from_numpy(np.stack([pair["rejected_tiles"] for pair in pairs])).long().unsqueeze(1),
            "pairs": [
                {key: value for key, value in pair.items() if key not in {"preferred_tiles", "rejected_tiles"}}
                for pair in pairs
            ],
            "candidate_count": int(len(self.candidates)),
            "total_seen": int(self.total_seen),
            "min_score_margin": float(min_score_margin),
        }
        return atomic_torch_save(payload, path)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "total_seen": self.total_seen,
            "rng_state": self.rng.bit_generator.state,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.max_candidates = int(max(0, payload.get("max_candidates", self.max_candidates)))
        self.total_seen = int(max(0, payload.get("total_seen", 0)))
        self.candidates = [PreferenceCandidate(**dict(item)) for item in payload.get("candidates", [])]
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, Mapping):
            self.rng.bit_generator.state = dict(rng_state)


__all__ = [
    "PreferenceCandidate",
    "QDPreferenceBuffer",
    "condition_graph_fingerprint",
    "deserialize_condition_graph",
    "serialize_condition_graph",
]
