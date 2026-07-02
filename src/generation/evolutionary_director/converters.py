"""MissionGraph/NetworkX conversion helpers."""

from __future__ import annotations

from ._shared import *
from ._shared import _stable_bidirectional_pair_key

def mission_graph_to_networkx(
    graph: MissionGraph,
    *,
    directed: bool = True,
) -> nx.Graph:
    """
    Convert MissionGraph to NetworkX graph for compatibility.
    
    VGLC Compliance: Preserves composite node labels and supports virtual nodes.
    
    Args:
        graph: MissionGraph from grammar
        
    Args:
        directed: Preserve edge directionality. Mission/progression semantics
            are directional; keep this True for solver/validator pipelines.

    Returns:
        NetworkX Graph/DiGraph with node attributes (may include VGLC virtual nodes)
    """
    G = nx.DiGraph() if directed else nx.Graph()
    if hasattr(graph, "generation_stats"):
        G.graph["generation_stats"] = copy.deepcopy(getattr(graph, "generation_stats", {}))
    
    # Add nodes with attributes
    for node_id, node in graph.nodes.items():
        # VGLC Compliance: Generate composite label if node has multiple attributes
        label = node.node_type.name  # Default: single label
        
        G.add_node(
            node_id,
            label=label,              # VGLC composite label
            type=node.node_type.name,
            difficulty=node.difficulty,
            position=node.position,
            key_id=node.key_id,
            required_item=node.required_item,
            item_type=node.item_type,
            switch_id=node.switch_id,
            is_hub=bool(node.is_hub),
            is_secret=bool(node.is_secret),
            room_size=tuple(node.room_size),
            sector_id=int(node.sector_id),
            sector_theme=node.sector_theme,
            virtual_layer=int(node.virtual_layer),
            is_arena=bool(node.is_arena),
            is_big_room=bool(node.is_big_room),
            token_id=node.token_id,
            difficulty_rating=node.difficulty_rating,
            is_sanctuary=bool(node.is_sanctuary),
            drops_resource=node.drops_resource,
            is_tutorial=bool(node.is_tutorial),
            is_mini_boss=bool(node.is_mini_boss),
            tension_value=float(node.tension_value),
            enemy_count_hint=int(max(0, int(getattr(node, "enemy_count_hint", 0) or 0))),
            key_count_hint=int(max(0, int(getattr(node, "key_count_hint", 0) or 0))),
            enemy_count=int(max(0, int(getattr(node, "enemy_count_hint", 0) or 0))),
            key_count=int(max(0, int(getattr(node, "key_count_hint", 0) or 0))),
            puzzle_count=int(
                max(
                    0,
                    int(
                        getattr(node, "puzzle_count_hint", 0)
                        or (
                            1
                            if node.node_type in {
                                NodeType.PUZZLE,
                                NodeType.TUTORIAL_PUZZLE,
                                NodeType.COMBAT_PUZZLE,
                                NodeType.COMPLEX_PUZZLE,
                            }
                            else 0
                        )
                    ),
                )
            ),
            item_count=int(
                max(
                    0,
                    int(
                        getattr(node, "item_count_hint", 0)
                        or (
                            1
                            if node.node_type in {
                                NodeType.ITEM,
                                NodeType.TREASURE,
                                NodeType.PROTECTION_ITEM,
                            }
                            else 0
                        )
                    ),
                )
            ),
            has_enemy=bool(
                int(getattr(node, "enemy_count_hint", 0) or 0) > 0
                or node.node_type in {NodeType.ENEMY, NodeType.BOSS, NodeType.MINI_BOSS, NodeType.ARENA, NodeType.COMBAT_PUZZLE}
            ),
            has_key=bool(
                int(getattr(node, "key_count_hint", 0) or 0) > 0
                or node.node_type in {NodeType.KEY, NodeType.BIG_KEY}
            ),
            has_puzzle=bool(
                int(getattr(node, "puzzle_count_hint", 0) or 0) > 0
                or node.node_type
                in {NodeType.PUZZLE, NodeType.TUTORIAL_PUZZLE, NodeType.COMBAT_PUZZLE, NodeType.COMPLEX_PUZZLE}
            ),
            has_item=bool(
                int(getattr(node, "item_count_hint", 0) or 0) > 0
                or node.node_type in {NodeType.ITEM, NodeType.TREASURE, NodeType.PROTECTION_ITEM}
            ),
        )
    
    # Preserve internal traversal semantics on output.
    # MissionGraph stores PATH edges once but treats them as bidirectional in
    # adjacency/pathfinding; export mirrored arcs so downstream directed graph
    # metrics match generation-time logic.
    bidirectional_output_types = set(graph.BIDIRECTIONAL_EDGE_TYPES)

    # Add edges
    for edge in graph.edges:
        source_node = graph.nodes.get(edge.source)
        target_node = graph.nodes.get(edge.target)
        is_goal_gauntlet_edge = bool(
            source_node is not None
            and target_node is not None
            and (
                (
                    source_node.node_type == NodeType.BOSS_DOOR
                    and target_node.node_type == NodeType.BOSS
                )
                or (
                    source_node.node_type == NodeType.BOSS
                    and target_node.node_type == NodeType.GOAL
                )
                or (
                    edge.edge_type == EdgeType.BOSS_LOCKED
                    and target_node.node_type == NodeType.BOSS_DOOR
                )
            )
        )
        edge_attrs = dict(
            label=edge.edge_type.name.lower(),  # VGLC edge label
            edge_type=edge.edge_type.name,
            key_required=edge.key_required,
            item_required=edge.item_required,
            switch_id=edge.switch_id,
            metadata=copy.deepcopy(edge.metadata),
            requires_key_count=int(edge.requires_key_count),
            token_count=int(edge.token_count),
            token_id=edge.token_id,
            is_window=bool(edge.is_window),
            hazard_damage=int(edge.hazard_damage),
            protection_item_id=edge.protection_item_id,
            preferred_direction=edge.preferred_direction,
            battery_id=edge.battery_id,
            switches_required=list(edge.switches_required or []),
            path_savings=int(edge.path_savings),
        )
        G.add_edge(edge.source, edge.target, **edge_attrs)
        if (
            directed
            and edge.edge_type in bidirectional_output_types
            and not is_goal_gauntlet_edge
            and (target_node is None or target_node.node_type != NodeType.GOAL)
            and not G.has_edge(edge.target, edge.source)
        ):
            reverse_attrs = dict(edge_attrs)
            metadata = reverse_attrs.get("metadata")
            if isinstance(metadata, dict):
                reverse_meta = copy.deepcopy(metadata)
                reverse_meta.setdefault("implied_reverse", True)
                reverse_attrs["metadata"] = reverse_meta
            else:
                reverse_attrs["metadata"] = {"implied_reverse": True}
            G.add_edge(edge.target, edge.source, **reverse_attrs)
    
    return G


def networkx_to_mission_graph(
    G: nx.Graph,
    *,
    assume_undirected_bidirectional: bool = True,
) -> MissionGraph:
    """
    Convert NetworkX graph back to MissionGraph.
    
    Args:
        G: NetworkX graph with node attributes
        
    Returns:
        MissionGraph
    """
    from src.generation.grammar import MissionNode
    
    graph = MissionGraph()
    stats = G.graph.get("generation_stats", {})
    if isinstance(stats, dict):
        graph.generation_stats.update({str(k): v for k, v in stats.items()})
        graph.ensure_generation_stats_defaults()
    
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value) != 0
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "1", "yes", "y", "on"}:
                return True
            if token in {"false", "0", "no", "n", "off", ""}:
                return False
        return bool(value)

    def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def _as_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)

    def _as_room_size(value: Any) -> Tuple[int, int]:
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return (int(value[0]), int(value[1]))
        return (1, 1)

    def _as_position(value: Any) -> Tuple[int, int, int]:
        if isinstance(value, (tuple, list)):
            if len(value) >= 3:
                return (int(value[0]), int(value[1]), int(value[2]))
            if len(value) == 2:
                return (int(value[0]), int(value[1]), 0)
        return (0, 0, 0)

    def _as_list_of_ints(value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, (tuple, list)):
            out: List[int] = []
            for v in value:
                iv = _as_int(v, None)
                if iv is not None:
                    out.append(int(iv))
            return out
        iv = _as_int(value, None)
        return [int(iv)] if iv is not None else []

    def _coerce_count_with_presence(
        data: Dict[str, Any],
        count_keys: Tuple[str, ...],
        presence: bool,
    ) -> int:
        for key in count_keys:
            parsed = _as_int(data.get(key), None)
            if parsed is not None:
                return max(0, int(parsed))
        return 1 if presence else 0

    def _infer_node_type_from_attrs(
        raw_type: str,
        label: str,
        data: Dict[str, Any],
    ) -> str:
        raw_upper = str(raw_type or "").upper().strip()
        if raw_upper in NodeType.__members__:
            return raw_upper

        tokens = set(parse_node_label_tokens(label))
        low_tokens = {t.lower() for t in tokens}

        is_start = bool({"s", "start"}.intersection(low_tokens) or "S" in tokens or _as_bool(data.get("is_start", False)))
        is_goal = bool({"t", "goal", "triforce"}.intersection(low_tokens) or _as_bool(data.get("is_goal", False)) or _as_bool(data.get("is_triforce", False)))
        is_boss = bool({"b", "boss"}.intersection(low_tokens) or _as_bool(data.get("is_boss", False)))
        is_key = bool({"k", "key", "small_key", "key_small"}.intersection(low_tokens))
        is_big_key = bool({"boss_key", "k"}.intersection(low_tokens) and "K" in tokens)
        is_item = bool({"i", "item", "minor_item", "macro_item", "key_item"}.intersection(low_tokens) or "I" in tokens)
        is_puzzle = bool({"p", "puzzle"}.intersection(low_tokens))
        is_enemy = bool({"e", "enemy"}.intersection(low_tokens))
        is_mini_boss = bool({"m", "miniboss", "mini_boss"}.intersection(low_tokens))
        is_switch = bool("s1" in low_tokens or _as_bool(data.get("has_switch", False)))

        if is_start:
            return "START"
        if is_goal:
            return "GOAL"
        if is_boss:
            return "BOSS"
        if is_big_key:
            return "BIG_KEY"
        if is_key:
            return "KEY"
        if is_switch:
            return "SWITCH"
        if is_item:
            return "ITEM"
        if is_puzzle:
            return "PUZZLE"
        if is_mini_boss:
            return "MINI_BOSS"
        if is_enemy:
            return "ENEMY"
        return "EMPTY"

    def _infer_edge_semantics(
        data: Dict[str, Any],
    ) -> Tuple[EdgeType, Optional[int], Optional[str], Optional[int], int, List[str], str]:
        label = str(data.get("label", "") or "")
        raw_edge_type = str(data.get("edge_type", data.get("type", "")) or "").strip()
        raw_upper = raw_edge_type.upper()
        constraints = [str(t).strip().lower() for t in parse_edge_type_tokens(label=label, edge_type=raw_edge_type)]
        cset = set(constraints)

        if raw_upper in EdgeType.__members__:
            edge_type = EdgeType[raw_upper]
        elif "boss_locked" in cset:
            edge_type = EdgeType.BOSS_LOCKED
        elif "key_locked" in cset:
            edge_type = EdgeType.LOCKED
        elif ("item_locked" in cset) or ("bombable" in cset):
            edge_type = EdgeType.ITEM_GATE
        elif "switch" in cset or "switch_locked" in cset:
            edge_type = EdgeType.ON_OFF_GATE
        elif "soft_locked" in cset:
            edge_type = EdgeType.ONE_WAY
        elif "stair" in cset:
            edge_type = EdgeType.STAIRS
        else:
            edge_type = EdgeType.PATH

        key_required = _as_int(data.get("key_required", data.get("key_id")), None)
        requires_key_count = max(0, int(_as_int(data.get("requires_key_count"), 0) or 0))
        if edge_type == EdgeType.LOCKED and key_required is None and requires_key_count <= 0:
            # VGLC 'k' edges represent fungible small-key locks by default.
            requires_key_count = 1

        item_required = data.get("item_required")
        if edge_type == EdgeType.ITEM_GATE and not item_required:
            if "bombable" in cset:
                item_required = "BOMB"
            elif "item_locked" in cset:
                item_required = "ITEM"

        switch_id = _as_int(data.get("switch_id"), None)
        return (
            edge_type,
            key_required,
            item_required,
            switch_id,
            requires_key_count,
            constraints,
            raw_edge_type,
        )

    # Add nodes
    for node_id in G.nodes():
        data = G.nodes[node_id]
        raw_type = str(data.get('type', '') or '').strip()
        label = str(data.get('label', '') or '')
        node_type_name = _infer_node_type_from_attrs(raw_type=raw_type, label=label, data=data)

        label_tokens = set(parse_node_label_tokens(label))
        low_tokens = {t.lower() for t in label_tokens}
        has_enemy_flag = bool(_as_bool(data.get("has_enemy", False)) or ("e" in low_tokens) or ("enemy" in low_tokens) or ("b" in low_tokens))
        has_key_flag = bool(_as_bool(data.get("has_key", False)) or ("k" in low_tokens) or ("key" in low_tokens) or ("small_key" in low_tokens))
        has_puzzle_flag = bool(_as_bool(data.get("has_puzzle", False)) or ("p" in low_tokens) or ("puzzle" in low_tokens))
        has_item_flag = bool(
            _as_bool(data.get("has_item", False))
            or _as_bool(data.get("has_macro_item", False))
            or _as_bool(data.get("has_minor_item", False))
            or ("i" in low_tokens)
            or ("item" in low_tokens)
            or ("macro_item" in low_tokens)
            or ("minor_item" in low_tokens)
            or ("I" in label_tokens)
            or ("m" in low_tokens)
        )

        enemy_hint = _coerce_count_with_presence(
            data,
            ("enemy_count_hint", "enemy_count"),
            presence=has_enemy_flag or node_type_name in {"ENEMY", "BOSS", "MINI_BOSS", "ARENA", "COMBAT_PUZZLE"},
        )
        key_hint = _coerce_count_with_presence(
            data,
            ("key_count_hint", "key_count"),
            presence=has_key_flag or node_type_name in {"KEY", "BIG_KEY"},
        )
        puzzle_hint = _coerce_count_with_presence(
            data,
            ("puzzle_count_hint", "puzzle_count"),
            presence=has_puzzle_flag or node_type_name in {"PUZZLE", "TUTORIAL_PUZZLE", "COMBAT_PUZZLE", "COMPLEX_PUZZLE"},
        )
        item_hint = _coerce_count_with_presence(
            data,
            ("item_count_hint", "item_count"),
            presence=has_item_flag or node_type_name in {"ITEM", "TREASURE", "PROTECTION_ITEM"},
        )
        node = MissionNode(
            id=node_id,
            node_type=NodeType[node_type_name],
            position=_as_position(data.get('position', (0, 0, 0))),
            key_id=_as_int(data.get('key_id'), None),
            difficulty=_as_float(data.get('difficulty', 0.5), 0.5),
            required_item=data.get('required_item'),
            item_type=data.get('item_type'),
            switch_id=_as_int(data.get('switch_id'), None),
            is_hub=_as_bool(data.get('is_hub', False)),
            is_secret=_as_bool(data.get('is_secret', False)),
            room_size=_as_room_size(data.get('room_size', (1, 1))),
            sector_id=int(_as_int(data.get('sector_id'), 0) or 0),
            sector_theme=data.get('sector_theme'),
            virtual_layer=int(_as_int(data.get('virtual_layer'), 0) or 0),
            is_arena=_as_bool(data.get('is_arena', False)),
            is_big_room=_as_bool(data.get('is_big_room', False)),
            token_id=data.get('token_id'),
            difficulty_rating=str(data.get('difficulty_rating', 'MODERATE')),
            is_sanctuary=_as_bool(data.get('is_sanctuary', False)),
            drops_resource=data.get('drops_resource'),
            is_tutorial=_as_bool(data.get('is_tutorial', False)),
            is_mini_boss=_as_bool(data.get('is_mini_boss', False)),
            tension_value=_as_float(data.get('tension_value', 0.5), 0.5),
            enemy_count_hint=int(max(0, enemy_hint)),
            key_count_hint=int(max(0, key_hint)),
        )
        # Optional composite-label hints (kept dynamic to preserve backwards compatibility
        # with existing MissionNode constructors and tensor shapes).
        node.puzzle_count_hint = int(max(0, puzzle_hint))
        node.item_count_hint = int(max(0, item_hint))
        graph.add_node(node)

    def _add_edge_with_attrs(src: Any, tgt: Any, data: Dict[str, Any]) -> EdgeType:
        edge_type, key_required, item_required, switch_id, requires_key_count, constraints, raw_edge_type = _infer_edge_semantics(data)
        graph.add_edge(
            src,
            tgt,
            edge_type,
            key_required,
            item_required,
            switch_id,
        )
        if not graph.edges:
            return edge_type
        edge_obj = graph.edges[-1]
        metadata = data.get('metadata', {})
        merged_meta = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
        merged_meta.setdefault("vglc_constraints", list(constraints))
        merged_meta.setdefault("source_edge_type_raw", str(raw_edge_type))
        edge_obj.metadata = merged_meta
        edge_obj.requires_key_count = int(max(0, requires_key_count))
        edge_obj.token_count = max(0, int(_as_int(data.get('token_count'), 0) or 0))
        edge_obj.token_id = data.get('token_id')
        edge_obj.is_window = _as_bool(data.get('is_window', False))
        edge_obj.hazard_damage = max(0, int(_as_int(data.get('hazard_damage'), 0) or 0))
        edge_obj.protection_item_id = data.get('protection_item_id')
        edge_obj.preferred_direction = data.get('preferred_direction')
        edge_obj.battery_id = _as_int(data.get('battery_id'), None)
        edge_obj.switches_required = _as_list_of_ints(data.get('switches_required'))
        edge_obj.path_savings = max(0, int(_as_int(data.get('path_savings'), 0) or 0))
        return edge_type

    # Add edges
    # When MissionGraph was exported as directed with mirrored implied reverse
    # edges for bidirectional semantics, skip those synthetic reverse arcs so
    # round-trip conversion preserves original edge count.
    seen_bidirectional_pairs: Set[Tuple[Tuple[str, str], Tuple[str, str], str]] = set()
    for src, tgt, data in G.edges(data=True):
        payload = data if isinstance(data, dict) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        if bool(metadata.get("implied_reverse", False)):
            continue

        inferred_edge_type, _, _, _, _, _, _ = _infer_edge_semantics(payload)
        bidirectional_types = {
            EdgeType.PATH,
            EdgeType.SHORTCUT,
            EdgeType.WARP,
            EdgeType.STAIRS,
            EdgeType.HIDDEN,
        }
        if inferred_edge_type in bidirectional_types:
            key = _stable_bidirectional_pair_key(
                src,
                tgt,
                inferred_edge_type.name,
            )
            if key in seen_bidirectional_pairs:
                continue
            seen_bidirectional_pairs.add(key)

        edge_type = _add_edge_with_attrs(src, tgt, payload)
        # If source graph is undirected, model edges as bidirectional transitions.
        if (not G.is_directed()) and bool(assume_undirected_bidirectional):
            if edge_type != EdgeType.ONE_WAY:
                _add_edge_with_attrs(tgt, src, payload)

    graph.sanitize()
    
    return graph
