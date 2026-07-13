"""Genome-to-mission-graph executor."""

from __future__ import annotations

from ._shared import *
from ._shared import _SAFE_RULE_NAME_RE

class GraphGrammarExecutor:
    """
    Executes a genome (sequence of rule IDs) to produce a phenotype (graph).
    
    The executor applies rules sequentially, skipping invalid rules rather
    than rejecting the entire genome. This ensures every genome produces
    a valid (though possibly suboptimal) graph.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        use_full_rule_space: bool = False,
        rule_space: Optional[str] = None,
        max_lock_key_rules: int = 3,
        rule_weight_overrides: Optional[Dict[str, float]] = None,
        enforce_generation_constraints: bool = True,
        allow_candidate_repairs: bool = False,
    ):
        """
        Initialize executor with available grammar rules.
        
        Args:
            seed: Random seed for deterministic execution
            use_full_rule_space: Backward-compatible switch for the full rule
                set. ``rule_space`` takes precedence when supplied.
            rule_space: ``core`` for the legacy rules, ``full`` for graph-only
                research, or ``spatial`` for mechanics that the final-map
                compiler and tile oracle can represent faithfully.
            max_lock_key_rules: Hard cap on progression key/lock-style rule
                applications allowed per genome execution.
            enforce_generation_constraints: Reject rule outcomes that violate
                lock/progression constraints at generation-time.
            allow_candidate_repairs: If True, try repairing an invalid candidate
                before rejecting it (kept off by default to reduce repair reliance).
        """
        self.seed = seed
        self.rng = random.Random(seed)
        requested_rule_space = (
            str(rule_space).strip().lower()
            if rule_space is not None
            else ("full" if use_full_rule_space else "core")
        )
        if requested_rule_space not in {"core", "full", "spatial"}:
            raise ValueError(
                f"Unsupported grammar rule_space={rule_space!r}; expected core, spatial, or full."
            )
        self.rule_space = requested_rule_space
        self.use_full_rule_space = requested_rule_space in {"full", "spatial"}
        self.max_lock_key_rules = int(max(0, max_lock_key_rules))
        self.rule_weight_overrides = rule_weight_overrides or {}
        self.enforce_generation_constraints = bool(enforce_generation_constraints)
        self.allow_candidate_repairs = bool(allow_candidate_repairs)
        
        if self.use_full_rule_space:
            # Reuse canonical grammar rule registry so evolutionary search
            # can explore the same topology mechanics as direct generation.
            canonical = MissionGrammar(seed=seed)
            if self.rule_space == "spatial":
                spatial_rule_names = {
                    "Start",
                    "InsertChallenge_ENEMY",
                    "InsertChallenge_PUZZLE",
                    "InsertLockKey",
                    "Branch",
                    "MergeShortcut",
                    "AddBossGauntlet",
                    "CreateHub",
                    "AddSecret",
                    "PruneGraph",
                    "AddFungibleLock",
                    "FormBigRoom",
                    "AddForeshadowing",
                    "AddSector",
                    "AddHazardGate",
                    "AddSkillChain",
                    "AddPacingBreaker",
                    "PruneDeadEnd",
                }
                self.rules = [
                    rule for rule in canonical.rules if rule.name in spatial_rule_names
                ]
            else:
                self.rules = canonical.rules
            self.rule_names = [rule.name for rule in self.rules]
        else:
            # Define available rules (indexed by ID)
            self.rules = [
                StartRule(),                              # 0
                InsertChallengeRule(NodeType.ENEMY),     # 1
                InsertChallengeRule(NodeType.PUZZLE),    # 2
                InsertLockKeyRule(),                      # 3
                BranchRule(),                             # 4
            ]
            
            # Map rule IDs to names for debugging
            self.rule_names = [
                "Start",
                "InsertChallenge_ENEMY",
                "InsertChallenge_PUZZLE",
                "InsertLockKey",
                "Branch",
            ]

        # Constraint helper used when full rule-space mode is enabled.
        self._constraint_grammar = MissionGrammar(seed=seed) if self.use_full_rule_space else None
        if self.rule_weight_overrides:
            for rule in self.rules:
                if rule.name in self.rule_weight_overrides:
                    try:
                        rule.weight = float(max(0.0, self.rule_weight_overrides[rule.name]))
                    except (TypeError, ValueError, OverflowError) as e:
                        logger.warning(
                            "Ignoring invalid rule weight override for '%s': %s",
                            rule.name,
                            e,
                        )
                        continue

    @staticmethod
    def _compute_node_degrees(graph: MissionGraph) -> Dict[Any, int]:
        """Compute undirected degree counts directly from MissionGraph edges."""
        deg: Dict[Any, int] = {nid: 0 for nid in graph.nodes.keys()}
        for edge in graph.edges:
            src = edge.source
            dst = edge.target
            deg[src] = int(deg.get(src, 0)) + 1
            if dst != src:
                deg[dst] = int(deg.get(dst, 0)) + 1
        return deg

    @staticmethod
    def _is_edge_expanding_rule_name(rule_name: str) -> bool:
        """Heuristic classifier for rules that can increase node connectivity."""
        name = str(rule_name or "").strip().lower()
        edge_keywords = (
            "branch",
            "merge",
            "lock",
            "gate",
            "shortcut",
            "teleport",
            "valve",
            "loop",
            "split",
            "hub",
            "sector",
            "stairs",
            "switch",
            "foreshadow",
            "arena",
        )
        return any(k in name for k in edge_keywords)

    @staticmethod
    def _is_lock_key_pressure_rule_name(rule_name: str) -> bool:
        """Classify rules that add progression gates or key-like resources."""
        name = str(rule_name or "").strip().lower()
        exact = {
            "insertlockkey",
            "addfungiblelock",
            "addmultilock",
            "addbossgauntlet",
            "addcollectionchallenge",
        }
        compact = name.replace("_", "").replace("-", "").replace(" ", "")
        if compact in exact:
            return True
        return any(
            token in name
            for token in (
                "lock",
                "key",
                "multi_lock",
                "multilock",
                "collection",
            )
        )

    @staticmethod
    def _estimate_rule_node_delta(rule_name: str) -> int:
        """Conservative node growth estimate for masking against max_nodes."""
        name = str(rule_name or "").strip().lower()
        if "start" in name:
            return 3
        if "branch" in name or "split" in name or "hub" in name or "sector" in name:
            return 2
        if "merge" in name or "prune" in name:
            return 0
        if "teleport" in name or "shortcut" in name or "valve" in name:
            return 1
        return 1

    def _build_action_mask(
        self,
        graph: MissionGraph,
        context: Dict[str, Any],
        *,
        max_nodes: int,
        allow_override: bool,
        lock_key_rule_count: int,
    ) -> Dict[int, bool]:
        """Dynamic admissible action mask for current graph state."""
        mask: Dict[int, bool] = {}
        degrees = self._compute_node_degrees(graph)
        has_degree_capacity = any(int(d) < 4 for d in degrees.values()) if degrees else True
        node_count = int(len(graph.nodes))

        for rid in range(1, len(self.rules)):
            rule = self.rules[rid]
            rule_name = self.rule_names[rid]

            if (
                self._is_lock_key_pressure_rule_name(rule_name)
                and lock_key_rule_count >= self.max_lock_key_rules
            ):
                mask[rid] = False
                continue

            if (not allow_override) and (node_count + self._estimate_rule_node_delta(rule_name) > int(max_nodes)):
                mask[rid] = False
                continue

            if self._is_edge_expanding_rule_name(rule_name) and (not has_degree_capacity):
                mask[rid] = False
                continue

            try:
                mask[rid] = bool(rule.can_apply(graph, context))
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError):
                mask[rid] = False

        return mask

    def _enforce_max_degree(self, graph: MissionGraph, max_degree: int = 4) -> int:
        """Deterministically prune excess incident edges to keep degree <= max_degree."""
        removed = 0
        if max_degree < 1 or not graph.edges:
            return removed

        # Stable pruning order: prefer pruning later-added soft/path edges first
        # and preserve progression-defining gates until there is no safer option.
        def _edge_prune_priority(item: Tuple[int, MissionEdge]) -> Tuple[int, int, int]:
            idx, edge = item
            edge_type_name = str(getattr(edge.edge_type, "name", edge.edge_type)).upper()
            is_soft_edge = int(edge_type_name not in {"PATH", "SHORTCUT", "HIDDEN", "WARP", "STAIRS"})
            is_progression_edge = int(
                edge_type_name in {
                    "LOCKED",
                    "BOSS_LOCKED",
                    "ITEM_GATE",
                    "ON_OFF_GATE",
                    "STATE_BLOCK",
                    "MULTI_LOCK",
                    "SHUTTER",
                    "HAZARD",
                    "ONE_WAY",
                }
            )
            # Lower sort keys are pruned first.
            return (is_progression_edge, is_soft_edge, -int(idx))

        changed = True
        while changed:
            changed = False
            deg = self._compute_node_degrees(graph)
            offenders = {nid for nid, d in deg.items() if int(d) > int(max_degree)}
            if not offenders:
                break

            ordered_edges = [
                edge
                for _, edge in sorted(enumerate(list(graph.edges)), key=_edge_prune_priority)
            ]
            for e in ordered_edges:
                if e.source in offenders or e.target in offenders:
                    try:
                        graph.edges.remove(e)
                        removed += 1
                        changed = True
                        break
                    except ValueError:
                        continue
            if changed:
                graph.sanitize()

        return removed
    
    def execute(
        self,
        genome: List[int],
        difficulty: float = 0.5,
        max_nodes: int = 20,
        allow_override: bool = False,
        record_trace: bool = False,
    ) -> MissionGraph:
        """
        Execute genome to produce a graph phenotype.
        
        Process:
        1. Apply StartRule to create initial graph
        2. For each rule_id in genome:
            - Check if rule is applicable
            - If yes, apply rule and update graph
            - If no, skip and continue
        3. Return final graph
        
        Args:
            genome: Sequence of rule IDs to execute
            difficulty: Base difficulty value (0.0-1.0)
            max_nodes: Maximum nodes to prevent explosion
            
        Returns:
            MissionGraph phenotype
        """
        graph = MissionGraph()
        
        context = {
            'rng': self.rng,
            'difficulty': difficulty,
            'goal_row': 5,
            'goal_col': 5,
            'spatial_compilable': self.rule_space == "spatial",
        }
        
        # Always apply start rule first
        graph = self.rules[0].apply(graph, context)
        graph.ensure_generation_stats_defaults()
        graph.generation_stats["require_goal_gauntlet"] = bool(self.use_full_rule_space)
        
        # Track statistics
        rules_applied = 0
        rules_skipped = 0
        lock_key_rule_count = 0
        lock_key_rule_skips = 0
        generation_constraint_rejections = 0
        candidate_repairs_applied = 0
        rule_trace: List[Dict[str, Any]] = []

        # Execute genome
        for genome_index, requested_rule_id in enumerate(genome):
            graph.sanitize()
            before_nodes = int(len(graph.nodes))
            before_edges = int(len(graph.edges))
            # Soft node cap: allow temporary overshoot so the EA can explore
            # through intermediate over-sized topologies to reach complex valid ones.
            # The fitness function's _constraint_violation() applies a smooth penalty
            # for node_count > max_nodes_soft. Hard safety valve at 1.5x prevents runaway.
            hard_cap = int(max(max_nodes + 2, max_nodes * 1.5))
            if len(graph.nodes) >= hard_cap:
                if record_trace:
                    rule_trace.append(
                        {
                            "genome_index": int(genome_index),
                            "requested_rule_id": int(requested_rule_id),
                            "rule_id": int(max(1, min(requested_rule_id, len(self.rules) - 1))),
                            "rule_name": str(
                                self.rule_names[max(1, min(requested_rule_id, len(self.rule_names) - 1))]
                            ),
                            "status": "stopped_hard_cap",
                            "reason": f"hard safety cap reached ({len(graph.nodes)} >= {hard_cap})",
                            "nodes_before": before_nodes,
                            "edges_before": before_edges,
                            "nodes_after": before_nodes,
                            "edges_after": before_edges,
                        }
                    )
                break
            
            # Clamp rule_id to valid range and build dynamic action mask.
            rule_id = max(1, min(requested_rule_id, len(self.rules) - 1))
            action_mask = self._build_action_mask(
                graph,
                context,
                max_nodes=max_nodes,
                allow_override=allow_override,
                lock_key_rule_count=lock_key_rule_count,
            )

            allowed_rule_ids = [rid for rid, allowed in action_mask.items() if allowed]
            requested_allowed = bool(action_mask.get(rule_id, False))

            if not requested_allowed:
                rules_skipped += 1
                if (
                    self._is_lock_key_pressure_rule_name(str(self.rule_names[rule_id]))
                    and lock_key_rule_count >= self.max_lock_key_rules
                ):
                    lock_key_rule_skips += 1
                if record_trace:
                    trace_row: Dict[str, Any] = {
                        "genome_index": int(genome_index),
                        "requested_rule_id": int(requested_rule_id),
                        "rule_id": int(rule_id),
                        "rule_name": str(self.rule_names[rule_id]),
                        "status": "skipped_action_masked",
                        "reason": (
                            "lock/key hard cap reached"
                            if (
                                self._is_lock_key_pressure_rule_name(str(self.rule_names[rule_id]))
                                and lock_key_rule_count >= self.max_lock_key_rules
                            )
                            else "requested action masked by dynamic feasibility constraints"
                        ),
                        "nodes_before": before_nodes,
                        "edges_before": before_edges,
                        "nodes_after": before_nodes,
                        "edges_after": before_edges,
                        "allowed_actions": [int(rid) for rid in allowed_rule_ids[:16]],
                    }
                    rule_trace.append(trace_row)
                continue

            rule = self.rules[rule_id]
            trace_row: Dict[str, Any] = {
                "genome_index": int(genome_index),
                "requested_rule_id": int(requested_rule_id),
                "rule_id": int(rule_id),
                "rule_name": str(self.rule_names[rule_id]),
                "status": "pending",
                "reason": "",
                "nodes_before": before_nodes,
                "edges_before": before_edges,
                "nodes_after": before_nodes,
                "edges_after": before_edges,
            }
            
            # Apply rule on a candidate copy so max-node constraints are enforced exactly.
            try:
                candidate = copy.deepcopy(graph)
                candidate = rule.apply(candidate, context)
                candidate.sanitize()

                if (not allow_override) and (len(candidate.nodes) > hard_cap):
                    rules_skipped += 1
                    if record_trace:
                        trace_row["status"] = "skipped_hard_cap"
                        trace_row["reason"] = (
                            f"candidate hard cap exceeded after apply ({len(candidate.nodes)} > {hard_cap})"
                        )
                        trace_row["nodes_after"] = int(len(candidate.nodes))
                        trace_row["edges_after"] = int(len(candidate.edges))
                        rule_trace.append(trace_row)
                    continue

                # Degree cap projection (keeps graph realizable without copy+reject churn).
                pruned_edges = self._enforce_max_degree(candidate, max_degree=4)

                # Optional strict progression enforcement in-place.
                if self._constraint_grammar is not None:
                    candidate = self._constraint_grammar.ensure_anchor_nodes(candidate)
                    candidate.sanitize()

                    lock_ok = bool(
                        self._constraint_grammar.validate_lock_key_ordering(
                            candidate,
                            log_failures=False,
                        )
                    )
                    prog_ok = bool(
                        self._constraint_grammar.validate_progression_constraints(
                            candidate,
                            log_failures=False,
                        )
                    )
                    if self.enforce_generation_constraints and (
                        not lock_ok or not prog_ok
                    ):
                        generation_constraint_rejections += 1
                        if self.allow_candidate_repairs:
                            candidate = self._constraint_grammar.fix_lock_key_ordering(candidate)
                            candidate.sanitize()
                            candidate = self._constraint_grammar.repair_progression_constraints(candidate)
                            candidate.sanitize()
                            candidate_repairs_applied += 1
                            lock_ok = bool(
                                self._constraint_grammar.validate_lock_key_ordering(
                                    candidate,
                                    log_failures=False,
                                )
                            )
                            prog_ok = bool(
                                self._constraint_grammar.validate_progression_constraints(
                                    candidate,
                                    log_failures=False,
                                )
                            )

                        if not lock_ok or not prog_ok:
                            rules_skipped += 1
                            if record_trace:
                                trace_row["status"] = "skipped_generation_constraints"
                                trace_row["reason"] = (
                                    "candidate violates generation constraints"
                                    + (" after repair attempt" if self.allow_candidate_repairs else "")
                                )
                                trace_row["nodes_after"] = int(len(candidate.nodes))
                                trace_row["edges_after"] = int(len(candidate.edges))
                                rule_trace.append(trace_row)
                            continue

                if (not allow_override) and (len(candidate.nodes) > hard_cap):
                    rules_skipped += 1
                    if record_trace:
                        trace_row["status"] = "skipped_hard_cap"
                        trace_row["reason"] = (
                            f"candidate hard cap exceeded after repair ({len(candidate.nodes)} > {hard_cap})"
                        )
                        trace_row["nodes_after"] = int(len(candidate.nodes))
                        trace_row["edges_after"] = int(len(candidate.edges))
                        rule_trace.append(trace_row)
                    continue

                graph = candidate

                rules_applied += 1
                if record_trace:
                    trace_row["status"] = "applied"
                    trace_row["nodes_after"] = int(len(graph.nodes))
                    trace_row["edges_after"] = int(len(graph.edges))
                    trace_row["pruned_edges_for_degree_cap"] = int(pruned_edges)
                    trace_row["lock_key_rule_count"] = int(
                        lock_key_rule_count
                        + (1 if self._is_lock_key_pressure_rule_name(str(self.rule_names[rule_id])) else 0)
                    )
                    rule_trace.append(trace_row)
                
                if self._is_lock_key_pressure_rule_name(str(self.rule_names[rule_id])):
                    lock_key_rule_count += 1
                    
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as error:
                logger.debug("Rule %s failed: %s", self.rule_names[rule_id], error)
                rules_skipped += 1
                if record_trace:
                    trace_row["status"] = "skipped_exception"
                    trace_row["reason"] = f"{type(error).__name__}: {error}"
                    rule_trace.append(trace_row)

        graph.ensure_generation_stats_defaults()
        graph.generation_stats["rule_applications"] = int(graph.generation_stats.get("rule_applications", 0)) + int(rules_applied + rules_skipped)
        graph.generation_stats["rule_applied"] = int(graph.generation_stats.get("rule_applied", 0)) + int(rules_applied)
        graph.generation_stats["rule_skipped"] = int(graph.generation_stats.get("rule_skipped", 0)) + int(rules_skipped)
        graph.generation_stats["generation_constraint_rejections"] = int(
            graph.generation_stats.get("generation_constraint_rejections", 0)
        ) + int(generation_constraint_rejections)
        graph.generation_stats["candidate_repairs_applied"] = int(
            graph.generation_stats.get("candidate_repairs_applied", 0)
        ) + int(candidate_repairs_applied)
        graph.generation_stats["lock_key_rule_applications"] = int(
            graph.generation_stats.get("lock_key_rule_applications", 0)
        ) + int(lock_key_rule_count)
        graph.generation_stats["lock_key_rule_cap_skips"] = int(
            graph.generation_stats.get("lock_key_rule_cap_skips", 0)
        ) + int(lock_key_rule_skips)
        if record_trace:
            graph.generation_stats["rule_trace"] = rule_trace
            graph.generation_stats["generation_replay"] = {
                "seed": self.seed,
                "difficulty": float(difficulty),
                "max_nodes": int(max_nodes),
                "allow_override": bool(allow_override),
                "rule_space": str(self.rule_space),
                "use_full_rule_space": bool(self.use_full_rule_space),
                "max_lock_key_rules": int(self.max_lock_key_rules),
                "enforce_generation_constraints": bool(self.enforce_generation_constraints),
                "allow_candidate_repairs": bool(self.allow_candidate_repairs),
                "rule_weight_overrides": {
                    str(k): float(v)
                    for k, v in dict(self.rule_weight_overrides).items()
                },
                "genome": [int(g) for g in genome],
                "rule_names": self.genome_to_rule_names(genome),
            }

        logger.debug(
            "Executed genome: %d applied, %d skipped, %d nodes, %d edges",
            rules_applied,
            rules_skipped,
            len(graph.nodes),
            len(graph.edges),
        )
        
        return graph
    
    def genome_to_rule_names(self, genome: List[int]) -> List[str]:
        """Convert genome to human-readable rule names."""
        names = []
        for rule_id in genome:
            rule_id = max(0, min(rule_id, len(self.rule_names) - 1))
            names.append(self.rule_names[rule_id])
        return names

    @classmethod
    def replay_from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        record_trace: bool = False,
        payload_size_budget: int = DEFAULT_REPLAY_PAYLOAD_MAX_BYTES,
    ) -> MissionGraph:
        """
        Rebuild a mission graph deterministically from serialized replay payload.
        """
        try:
            payload_budget = int(payload_size_budget)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError("Replay payload size budget must be an integer.") from e
        payload_budget = max(1024, payload_budget)

        def _bounded_int(
            field: str,
            value: Any,
            *,
            default: int,
            lo: int,
            hi: int,
        ) -> int:
            if value is None:
                return int(default)
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"Replay payload field '{field}' must be an integer.") from e
            if parsed < int(lo) or parsed > int(hi):
                raise ValueError(
                    f"Replay payload field '{field}' out of bounds: {parsed} not in [{lo}, {hi}]"
                )
            return parsed

        def _bounded_float(
            field: str,
            value: Any,
            *,
            default: float,
            lo: float,
            hi: float,
        ) -> float:
            if value is None:
                return float(default)
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"Replay payload field '{field}' must be a float.") from e
            if (not math.isfinite(parsed)) or parsed < float(lo) or parsed > float(hi):
                raise ValueError(
                    f"Replay payload field '{field}' out of bounds: {parsed} not in [{lo}, {hi}]"
                )
            return parsed

        def _bounded_bool(
            field: str,
            value: Any,
            *,
            default: bool,
        ) -> bool:
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                return bool(int(value))
            raise ValueError(f"Replay payload field '{field}' must be a boolean.")

        def _estimate_json_size(
            value: Any,
            *,
            depth: int = 0,
            budget: int = payload_budget,
        ) -> int:
            if depth > 16:
                raise ValueError("Replay payload nesting exceeds supported depth.")
            if value is None:
                return 4
            if isinstance(value, bool):
                return 4 if value else 5
            if isinstance(value, (int, np.integer)):
                return len(str(int(value)))
            if isinstance(value, (float, np.floating)):
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError("Replay payload contains non-finite numeric values.")
                return len(format(parsed, ".17g"))
            if isinstance(value, str):
                return len(value.encode("utf-8")) + 2
            if isinstance(value, (list, tuple)):
                total = 2
                for idx, item in enumerate(value):
                    if idx > 0:
                        total += 1
                    total += _estimate_json_size(item, depth=depth + 1, budget=budget)
                    if total > budget:
                        raise ValueError(
                            f"Replay payload too large ({total} bytes > {budget} bytes)."
                        )
                return total
            if isinstance(value, dict):
                total = 2
                for idx, (raw_key, raw_val) in enumerate(value.items()):
                    if idx > 0:
                        total += 1
                    if not isinstance(raw_key, (str, int, np.integer, float, np.floating, bool)):
                        raise ValueError("Replay payload keys must be JSON-compatible scalar values.")
                    total += _estimate_json_size(str(raw_key), depth=depth + 1, budget=budget)
                    total += 1
                    total += _estimate_json_size(raw_val, depth=depth + 1, budget=budget)
                    if total > budget:
                        raise ValueError(
                            f"Replay payload too large ({total} bytes > {budget} bytes)."
                        )
                return total
            raise ValueError(
                f"Replay payload contains unsupported value type: {type(value).__name__}."
            )

        def _sanitize_rule_name(
            raw_name: Any,
            *,
            field: str,
            index: Optional[int] = None,
        ) -> str:
            name = str(raw_name).strip()
            suffix = f" {index}" if index is not None else ""
            if not name or len(name) > 128:
                field_label = f"{field}s" if field == "override key" else field
                raise ValueError(
                    f"Replay payload {field_label}{suffix} must be 1-128 characters."
                )
            if not _SAFE_RULE_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"Replay payload {field}{suffix} must match {_SAFE_RULE_NAME_RE.pattern}."
                )
            return name

        if not isinstance(payload, dict):
            raise ValueError("Replay payload must be a dictionary.")
        allowed_fields = {
            "seed",
            "difficulty",
            "max_nodes",
            "allow_override",
            "rule_space",
            "use_full_rule_space",
            "max_lock_key_rules",
            "enforce_generation_constraints",
            "allow_candidate_repairs",
            "rule_weight_overrides",
            "genome",
            "rule_names",
        }
        unknown_fields = sorted(str(key) for key in payload.keys() if str(key) not in allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"Replay payload contains unknown fields: {', '.join(unknown_fields[:8])}"
            )
        payload_size = _estimate_json_size(payload, budget=payload_budget)
        if payload_size > payload_budget:
            raise ValueError(
                f"Replay payload too large ({payload_size} bytes > {payload_budget} bytes)."
            )
        raw_genome = payload.get("genome", [])
        if not isinstance(raw_genome, list):
            raise ValueError("Replay payload missing list field 'genome'.")
        if len(raw_genome) > 1000:
            raise ValueError(
                f"Replay payload genome too long ({len(raw_genome)} > 1000)."
            )
        genome: List[int] = []
        for idx, raw_rule in enumerate(raw_genome):
            try:
                rule_id = int(raw_rule)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"Replay payload genome entry {idx} is not an integer.") from e
            if abs(rule_id) > 100000:
                raise ValueError(
                    f"Replay payload genome entry {idx} is out of bounds: {rule_id}"
                )
            genome.append(rule_id)

        raw_overrides = payload.get("rule_weight_overrides", {})
        if raw_overrides is None:
            raw_overrides = {}
        if not isinstance(raw_overrides, dict):
            raise ValueError("Replay payload field 'rule_weight_overrides' must be a dictionary.")
        if len(raw_overrides) > 256:
            raise ValueError(
                f"Replay payload has too many rule weight overrides ({len(raw_overrides)} > 256)."
            )
        safe_overrides: Dict[str, float] = {}
        for raw_key, raw_value in raw_overrides.items():
            key = _sanitize_rule_name(raw_key, field="override key")
            try:
                weight = float(raw_value)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"Replay payload override '{key}' must be numeric.") from e
            if (not math.isfinite(weight)) or weight < 0.0 or weight > 100.0:
                raise ValueError(
                    f"Replay payload override '{key}' out of bounds: {weight} not in [0, 100]"
                )
            safe_overrides[key] = weight

        raw_rule_names = payload.get("rule_names")
        safe_rule_names: Optional[List[str]] = None
        if raw_rule_names is not None:
            if not isinstance(raw_rule_names, list):
                raise ValueError("Replay payload field 'rule_names' must be a list.")
            if len(raw_rule_names) > 1000:
                raise ValueError(
                    f"Replay payload field 'rule_names' too long ({len(raw_rule_names)} > 1000)."
                )
            safe_rule_names = []
            seen_rule_names: Set[str] = set()
            for idx, raw_name in enumerate(raw_rule_names):
                name = _sanitize_rule_name(raw_name, field="rule name", index=idx)
                if name in seen_rule_names:
                    raise ValueError(f"Replay payload rule name {idx} duplicates '{name}'.")
                seen_rule_names.add(name)
                safe_rule_names.append(name)

        replay_rule_space = payload.get("rule_space")
        if replay_rule_space is None:
            replay_rule_space = (
                "full"
                if _bounded_bool(
                    "use_full_rule_space",
                    payload.get("use_full_rule_space", False),
                    default=False,
                )
                else "core"
            )
        replay_rule_space = str(replay_rule_space).strip().lower()
        if replay_rule_space not in {"core", "spatial", "full"}:
            raise ValueError(
                "Replay payload field 'rule_space' must be core, spatial, or full."
            )

        executor = cls(
            seed=(
                None
                if payload.get("seed") is None
                else _bounded_int(
                    "seed",
                    payload.get("seed"),
                    default=0,
                    lo=-(2**31),
                    hi=(2**31) - 1,
                )
            ),
            use_full_rule_space=replay_rule_space in {"spatial", "full"},
            rule_space=replay_rule_space,
            max_lock_key_rules=_bounded_int(
                "max_lock_key_rules",
                payload.get("max_lock_key_rules", 3),
                default=3,
                lo=0,
                hi=128,
            ),
            rule_weight_overrides=safe_overrides,
            enforce_generation_constraints=_bounded_bool(
                "enforce_generation_constraints",
                payload.get("enforce_generation_constraints", True),
                default=True,
            ),
            allow_candidate_repairs=_bounded_bool(
                "allow_candidate_repairs",
                payload.get("allow_candidate_repairs", False),
                default=False,
            ),
        )
        if safe_overrides:
            valid_rule_names = set(executor.rule_names)
            unknown_override_names = sorted(name for name in safe_overrides.keys() if name not in valid_rule_names)
            if unknown_override_names:
                raise ValueError(
                    f"Replay payload override keys must reference known rules: {', '.join(unknown_override_names[:8])}"
                )
        if safe_rule_names is not None:
            valid_rule_names = set(executor.rule_names)
            unknown_rule_names = sorted(name for name in safe_rule_names if name not in valid_rule_names)
            if unknown_rule_names:
                raise ValueError(
                    f"Replay payload rule_names must reference known rules: {', '.join(unknown_rule_names[:8])}"
                )
        return executor.execute(
            genome=genome,
            difficulty=_bounded_float(
                "difficulty",
                payload.get("difficulty", 0.5),
                default=0.5,
                lo=0.0,
                hi=1.0,
            ),
            max_nodes=_bounded_int(
                "max_nodes",
                payload.get("max_nodes", 20),
                default=20,
                lo=1,
                hi=512,
            ),
            allow_override=_bounded_bool(
                "allow_override",
                payload.get("allow_override", False),
                default=False,
            ),
            record_trace=bool(record_trace),
        )
