from __future__ import annotations

import networkx as nx

from src.core.domain import ZeldaSchema, resolve_domain_schema
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline


def test_zelda_schema_preserves_existing_room_role_semantics() -> None:
    schema = ZeldaSchema()
    flags = schema.room_role_flags(
        {
            "label": "b,k,t",
            "type": "combat_puzzle",
            "difficulty_rating": "HARD",
        }
    )

    assert flags["has_boss"] is True
    assert flags["has_key"] is True
    assert flags["has_goal"] is True
    assert flags["has_puzzle"] is True
    assert flags["is_combat_puzzle"] is True
    assert flags["is_complex_puzzle"] is True


class CustomSchema:
    name = "custom_unit_test"

    def room_role_flags(self, attrs):
        return {
            "is_start": attrs.get("kind") == "spawn",
            "has_goal": attrs.get("kind") == "exit",
            "has_puzzle": attrs.get("kind") == "riddle",
        }


def test_pipeline_accepts_custom_domain_schema_for_graph_context_roles() -> None:
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        domain_schema=CustomSchema(),
    )
    graph = nx.DiGraph()
    graph.add_node("a", kind="spawn")
    graph.add_node("b", kind="exit")
    graph.add_edge("a", "b")

    graph_data = pipeline._prepare_graph_context(graph, use_tpe=True)

    assert pipeline.domain_schema_name == "custom_unit_test"
    assert graph_data["start_node_id"] == graph_data["node_to_idx"]["a"]
    assert graph_data["target_idx"] == graph_data["node_to_idx"]["b"]
    assert pipeline._room_role_flags({"kind": "riddle"})["has_puzzle"] is True
    assert pipeline._room_role_flags({"kind": "riddle"})["has_boss"] is False


def test_domain_schema_resolver_defaults_to_zelda() -> None:
    assert isinstance(resolve_domain_schema(None), ZeldaSchema)
    assert isinstance(resolve_domain_schema("zelda"), ZeldaSchema)
