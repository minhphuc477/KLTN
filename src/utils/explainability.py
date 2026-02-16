"""
Feature 9: Explainability System (Decision Tracing)
===================================================
Complete traceable lineage of all AI decisions for designer transparency.

Problem:
    Designer needs to understand:
    - WHY is there a bombable wall in Room 4?
    - WHY did fitness drop after mutation?
    - WHY did LogicNet mark this tile invalid?
    - WHICH grammar rule created this lock?
    
    Current system is a black box without decision provenance.

Solution:
    1. Rule Tagging: Tag every grammar rule application
    2. Fitness Attribution: Track fitness contribution of each mutation
    3. LogicNet Confidence: Export attention maps
    4. Genealogy Tracking: Full evolutionary lineage
    5. GUI Debug Overlay: Visual decision highlighting

Integration Point: Throughout pipeline, aggregated in GUI debug panel
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class DecisionSource(Enum):
    """Source of a design decision."""
    EVOLUTIONARY_DIRECTOR = "evolutionary_director"
    GRAMMAR_RULE = "grammar_rule"
    DIFFUSION_MODEL = "diffusion_model"
    LOGIC_NET = "logic_net"
    WFC_REFINER = "wfc_refiner"
    SYMBOLIC_REFINER = "symbolic_refiner"
    USER_CONSTRAINT = "user_constraint"


@dataclass
class DecisionTrace:
    """
    A single traceable decision in the pipeline.
    
    Example:
        DecisionTrace(
            decision_id="rule_apply_123",
            source=DecisionSource.GRAMMAR_RULE,
            timestamp=datetime.now(),
            description="Applied InsertLockKey(key_id=7) at node 4",
            confidence=0.87,
            parent_decision="mutation_456",
            affected_elements={"nodes": [4, 5], "edges": [(4, 5)]},
            metadata={"rule_type": "InsertLockKey", "key_id": 7}
        )
    """
    decision_id: str
    source: DecisionSource
    timestamp: datetime
    description: str
    confidence: float  # 0.0 (uncertain) to 1.0 (definite)
    parent_decision: Optional[str] = None  # For genealogy
    affected_elements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fitness_contribution: Optional[float] = None  # Δfitness from this decision
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "decision_id": self.decision_id,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "confidence": self.confidence,
            "parent_decision": self.parent_decision,
            "affected_elements": self.affected_elements,
            "metadata": self.metadata,
            "fitness_contribution": self.fitness_contribution
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionTrace":
        """Deserialize from JSON."""
        return cls(
            decision_id=data["decision_id"],
            source=DecisionSource(data["source"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data["description"],
            confidence=data["confidence"],
            parent_decision=data.get("parent_decision"),
            affected_elements=data.get("affected_elements", {}),
            metadata=data.get("metadata", {}),
            fitness_contribution=data.get("fitness_contribution")
        )


@dataclass
class GrammarRuleTrace(DecisionTrace):
    """Specialized trace for grammar rule applications."""
    rule_name: str = ""
    rule_parameters: Dict[str, Any] = field(default_factory=dict)
    nodes_added: List[int] = field(default_factory=list)
    edges_added: List[Tuple[int, int]] = field(default_factory=list)
    execution_order: int = 0  # Position in rule sequence


@dataclass
class TileDecisionTrace(DecisionTrace):
    """Specialized trace for tile-level decisions."""
    room_id: int = 0
    position: Tuple[int, int] = (0, 0)
    tile_id: int = 0
    alternatives_considered: List[int] = field(default_factory=list)
    logicnet_confidence: Optional[float] = None
    repair_applied: bool = False


@dataclass
class FitnessAttribution:
    """Attribution of fitness to specific decisions."""
    generation_id: int
    genome_id: str
    total_fitness: float
    decision_contributions: Dict[str, float]  # {decision_id: Δfitness}
    
    def get_top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N decisions by fitness contribution."""
        sorted_contrib = sorted(
            self.decision_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_contrib[:n]


@dataclass
class EvolutionaryGenealogy:
    """Full evolutionary lineage of a genome."""
    genome_id: str
    generation: int
    fitness: float
    parent_id: Optional[str]
    children_ids: List[str]
    mutations_applied: List[str]  # Decision IDs of mutations
    selection_reason: str  # Why this genome was selected
    
    def get_lineage(self, genealogy_db: Dict[str, "EvolutionaryGenealogy"]) -> List[str]:
        """Get full lineage from root to this genome."""
        lineage = [self.genome_id]
        current_id = self.parent_id
        
        while current_id is not None:
            lineage.append(current_id)
            current_id = genealogy_db[current_id].parent_id
        
        return list(reversed(lineage))


# ============================================================================
# EXPLAINABILITY MANAGER
# ============================================================================

class ExplainabilityManager:
    """
    Central manager for all decision traces and explanations.
    
    Stores complete provenance of:
    - Grammar rule applications
    - Evolutionary mutations
    - Neural network predictions
    - Symbolic repair decisions
    - User interventions
    
    Provides queries like:
    - "Why is there a lock in Room 4?"
    - "What caused fitness to drop?"
    - "Which rule created this edge?"
    - "Show me the genealogy of this genome"
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        self.traces: Dict[str, DecisionTrace] = {}
        self.genealogy: Dict[str, EvolutionaryGenealogy] = {}
        self.fitness_attributions: Dict[int, Dict[str, FitnessAttribution]] = {}  # {generation: {genome_id: attribution}}
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Index for fast lookups
        self.traces_by_element: Dict[str, List[str]] = {}  # {element_id: [decision_ids]}
        self.traces_by_source: Dict[DecisionSource, List[str]] = {src: [] for src in DecisionSource}
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def add_trace(self, trace: DecisionTrace):
        """Add a decision trace to the system."""
        self.traces[trace.decision_id] = trace
        
        # Update indices
        self.traces_by_source[trace.source].append(trace.decision_id)
        
        for element_type, elements in trace.affected_elements.items():
            if isinstance(elements, list):
                for elem in elements:
                    key = f"{element_type}_{elem}"
                    if key not in self.traces_by_element:
                        self.traces_by_element[key] = []
                    self.traces_by_element[key].append(trace.decision_id)
        
        logger.debug(f"Added trace: {trace.decision_id} ({trace.description})")
    
    def add_genealogy(self, genealogy: EvolutionaryGenealogy):
        """Add evolutionary genealogy record."""
        self.genealogy[genealogy.genome_id] = genealogy
    
    def add_fitness_attribution(self, generation: int, attribution: FitnessAttribution):
        """Add fitness attribution for a genome."""
        if generation not in self.fitness_attributions:
            self.fitness_attributions[generation] = {}
        self.fitness_attributions[generation][attribution.genome_id] = attribution
    
    # ========================================================================
    # QUERY INTERFACE
    # ========================================================================
    
    def why_node(self, node_id: int, mission_graph: nx.Graph) -> List[DecisionTrace]:
        """
        Answer: "Why does node {node_id} exist?"
        
        Returns all traces that affected this node.
        """
        key = f"nodes_{node_id}"
        if key not in self.traces_by_element:
            return []
        
        trace_ids = self.traces_by_element[key]
        traces = [self.traces[tid] for tid in trace_ids]
        
        # Sort by timestamp
        traces = sorted(traces, key=lambda t: t.timestamp)
        
        return traces
    
    def why_edge(self, source: int, target: int) -> List[DecisionTrace]:
        """Answer: "Why does edge (source, target) exist?" """
        key = f"edges_{(source, target)}"
        if key not in self.traces_by_element:
            return []
        
        trace_ids = self.traces_by_element[key]
        return [self.traces[tid] for tid in trace_ids]
    
    def why_tile(self, room_id: int, position: Tuple[int, int]) -> List[TileDecisionTrace]:
        """Answer: "Why is tile at position (row, col) in Room X this type?" """
        results = []
        
        for trace in self.traces.values():
            if isinstance(trace, TileDecisionTrace):
                if trace.room_id == room_id and trace.position == position:
                    results.append(trace)
        
        return sorted(results, key=lambda t: t.timestamp)
    
    def what_caused_fitness_change(
        self,
        genome_id: str,
        generation: int
    ) -> List[Tuple[str, float]]:
        """
        Answer: "What decisions caused fitness to change?"
        
        Returns list of (decision_id, Δfitness) sorted by impact.
        """
        if generation not in self.fitness_attributions:
            return []
        
        if genome_id not in self.fitness_attributions[generation]:
            return []
        
        attribution = self.fitness_attributions[generation][genome_id]
        return attribution.get_top_contributors(n=10)
    
    def get_genealogy_lineage(self, genome_id: str) -> List[EvolutionaryGenealogy]:
        """Get full evolutionary lineage for a genome."""
        if genome_id not in self.genealogy:
            return []
        
        lineage_ids = self.genealogy[genome_id].get_lineage(self.genealogy)
        return [self.genealogy[gid] for gid in lineage_ids]
    
    def explain_mutation(self, decision_id: str) -> Dict[str, Any]:
        """
        Explain a specific mutation decision.
        
        Returns detailed explanation including:
        - What changed
        - Why it was selected
        - Fitness impact
        - Alternative mutations considered
        """
        if decision_id not in self.traces:
            return {"error": f"Decision {decision_id} not found"}
        
        trace = self.traces[decision_id]
        
        explanation = {
            "decision_id": decision_id,
            "description": trace.description,
            "source": trace.source.value,
            "confidence": trace.confidence,
            "affected_elements": trace.affected_elements,
            "fitness_contribution": trace.fitness_contribution,
            "metadata": trace.metadata
        }
        
        # Find child decisions (consequences)
        children = [t for t in self.traces.values() if t.parent_decision == decision_id]
        explanation["consequences"] = [
            {"id": c.decision_id, "description": c.description}
            for c in children
        ]
        
        return explanation
    
    def get_decisions_by_source(self, source: DecisionSource) -> List[DecisionTrace]:
        """Get all decisions from a specific source (e.g., all LogicNet decisions)."""
        trace_ids = self.traces_by_source[source]
        return [self.traces[tid] for tid in trace_ids]
    
    # ========================================================================
    # EXPORT / PERSISTENCE
    # ========================================================================
    
    def export_json(self, filepath: str):
        """Export all traces to JSON file."""
        data = {
            "traces": {tid: trace.to_dict() for tid, trace in self.traces.items()},
            "genealogy": {
                gid: {
                    "genome_id": g.genome_id,
                    "generation": g.generation,
                    "fitness": g.fitness,
                    "parent_id": g.parent_id,
                    "children_ids": g.children_ids,
                    "mutations_applied": g.mutations_applied,
                    "selection_reason": g.selection_reason
                }
                for gid, g in self.genealogy.items()
            },
            "fitness_attributions": {
                gen: {
                    gid: {
                        "genome_id": attr.genome_id,
                        "total_fitness": attr.total_fitness,
                        "decision_contributions": attr.decision_contributions
                    }
                    for gid, attr in gen_data.items()
                }
                for gen, gen_data in self.fitness_attributions.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported explainability data to {filepath}")
    
    def load_json(self, filepath: str):
        """Load traces from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load traces
        for tid, trace_data in data.get("traces", {}).items():
            trace = DecisionTrace.from_dict(trace_data)
            self.add_trace(trace)
        
        # Load genealogy
        for gid, gen_data in data.get("genealogy", {}).items():
            genealogy = EvolutionaryGenealogy(**gen_data)
            self.add_genealogy(genealogy)
        
        logger.info(f"Loaded explainability data from {filepath}")
    
    def generate_html_report(self, output_path: str):
        """Generate interactive HTML report of decision traces."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>KLTN Explainability Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .trace {{ border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .trace-header {{ font-weight: bold; color: #1a73e8; }}
        .trace-meta {{ color: #666; font-size: 0.9em; }}
        .fitness-positive {{ color: green; }}
        .fitness-negative {{ color: red; }}
        .genealogy-tree {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>KLTN Decision Tracing Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    
    <h2>Summary</h2>
    <ul>
        <li>Total Decisions: {len(self.traces)}</li>
        <li>Generations Tracked: {len(self.fitness_attributions)}</li>
        <li>Genomes in Genealogy: {len(self.genealogy)}</li>
    </ul>
    
    <h2>Decisions by Source</h2>
    """
        
        for source, trace_ids in self.traces_by_source.items():
            if not trace_ids:
                continue
            
            html += f"<h3>{source.value.replace('_', ' ').title()} ({len(trace_ids)} decisions)</h3>"
            
            for tid in trace_ids[:20]:  # Show first 20
                trace = self.traces[tid]
                html += f"""
                <div class="trace">
                    <div class="trace-header">{trace.description}</div>
                    <div class="trace-meta">
                        ID: {trace.decision_id} | 
                        Confidence: {trace.confidence:.2f} | 
                        Timestamp: {trace.timestamp.strftime("%H:%M:%S")}
                    </div>
                """
                
                if trace.fitness_contribution is not None:
                    color_class = "fitness-positive" if trace.fitness_contribution > 0 else "fitness-negative"
                    html += f'<div class="{color_class}">Fitness Δ: {trace.fitness_contribution:+.3f}</div>'
                
                html += "</div>"
        
        html += """
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report: {output_path}")


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def trace_grammar_rule_application(
    explainability_manager: ExplainabilityManager,
    rule_name: str,
    genome_id: str,
    nodes_added: List[int],
    edges_added: List[Tuple[int, int]],
    execution_order: int,
    parent_decision_id: Optional[str] = None
) -> str:
    """
    Trace a grammar rule application.
    
    Call this in EvolutionaryDirector every time a rule is applied.
    """
    decision_id = f"grammar_{genome_id}_{rule_name}_{execution_order}"
    
    trace = GrammarRuleTrace(
        decision_id=decision_id,
        source=DecisionSource.GRAMMAR_RULE,
        timestamp=datetime.now(),
        description=f"Applied {rule_name} (added {len(nodes_added)} nodes, {len(edges_added)} edges)",
        confidence=1.0,  # Grammar rules are deterministic
        parent_decision=parent_decision_id,
        affected_elements={
            "nodes": nodes_added,
            "edges": edges_added
        },
        metadata={"genome_id": genome_id},
        rule_name=rule_name,
        nodes_added=nodes_added,
        edges_added=edges_added,
        execution_order=execution_order
    )
    
    explainability_manager.add_trace(trace)
    return decision_id


def trace_tile_decision(
    explainability_manager: ExplainabilityManager,
    room_id: int,
    position: Tuple[int, int],
    tile_id: int,
    source: DecisionSource,
    confidence: float,
    logicnet_confidence: Optional[float] = None,
    repair_applied: bool = False,
    alternatives: Optional[List[int]] = None
) -> str:
    """
    Trace a tile-level decision.
    
    Call this in SymbolicRefiner or LogicNet when placing/repairing tiles.
    """
    decision_id = f"tile_{room_id}_{position[0]}_{position[1]}_{tile_id}"
    
    trace = TileDecisionTrace(
        decision_id=decision_id,
        source=source,
        timestamp=datetime.now(),
        description=f"Room {room_id} position {position}: tile_id={tile_id}",
        confidence=confidence,
        affected_elements={"room": [room_id], "position": [position]},
        metadata={"tile_id": tile_id},
        room_id=room_id,
        position=position,
        tile_id=tile_id,
        alternatives_considered=alternatives or [],
        logicnet_confidence=logicnet_confidence,
        repair_applied=repair_applied
    )
    
    explainability_manager.add_trace(trace)
    return decision_id


def compute_fitness_attribution(
    explainability_manager: ExplainabilityManager,
    genome_id: str,
    generation: int,
    fitness: float,
    mutations_applied: List[str],  # Decision IDs
    fitness_deltas: Dict[str, float]  # {decision_id: Δfitness}
):
    """
    Compute and store fitness attribution.
    
    Call this after evaluating a genome to track which mutations helped/hurt.
    """
    attribution = FitnessAttribution(
        generation_id=generation,
        genome_id=genome_id,
        total_fitness=fitness,
        decision_contributions=fitness_deltas
    )
    
    explainability_manager.add_fitness_attribution(generation, attribution)
    
    # Update fitness_contribution in traces
    for decision_id, delta in fitness_deltas.items():
        if decision_id in explainability_manager.traces:
            explainability_manager.traces[decision_id].fitness_contribution = delta


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Trace a simple evolutionary run
    
    manager = ExplainabilityManager(log_dir="outputs/explainability")
    
    # Generation 0: Initial genome
    genome_0 = "genome_0_gen_0"
    
    # Trace rule applications
    d1 = trace_grammar_rule_application(
        manager,
        rule_name="StartRule",
        genome_id=genome_0,
        nodes_added=[0],
        edges_added=[],
        execution_order=0
    )
    
    d2 = trace_grammar_rule_application(
        manager,
        rule_name="InsertChallenge_ENEMY",
        genome_id=genome_0,
        nodes_added=[1],
        edges_added=[(0, 1)],
        execution_order=1,
        parent_decision_id=d1
    )
    
    # Add genealogy
    manager.add_genealogy(EvolutionaryGenealogy(
        genome_id=genome_0,
        generation=0,
        fitness=0.75,
        parent_id=None,
        children_ids=[],
        mutations_applied=[d1, d2],
        selection_reason="Initial random genome"
    ))
    
    # Query explanations
    print("=== Why does node 1 exist? ===")
    traces = manager.why_node(1, None)
    for trace in traces:
        print(f"  - {trace.description}")
    
    # Export
    manager.export_json("outputs/explainability/traces.json")
    manager.generate_html_report("outputs/explainability/report.html")
    
    print(f"\n✓ Explainability data saved. Open outputs/explainability/report.html")
