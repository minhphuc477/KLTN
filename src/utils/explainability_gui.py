"""
GUI Debug Overlay for Explainability System
============================================
Visual highlighting of AI decisions in the GUI for designer transparency.

Integrates with gui_runner.py to show:
- Why each node exists (hover tooltip)
- Fitness contribution heatmap
- Genealogy tree visualization
- Tile decision confidence overlay
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

from src.utils.explainability import (
    ExplainabilityManager,
    DecisionTrace,
    DecisionSource,
    EvolutionaryGenealogy
)

# ============================================================================
# DEBUG OVERLAY RENDERER
# ============================================================================

@dataclass
class OverlayConfig:
    """Configuration for debug overlay visualization."""
    show_decision_tooltips: bool = True
    show_fitness_heatmap: bool = True
    show_genealogy_tree: bool = False
    show_confidence_overlay: bool = True
    overlay_alpha: float = 0.7
    tooltip_font_size: int = 14
    heatmap_colormap: str = 'coolwarm'


class ExplainabilityDebugOverlay:
    """
    Renders explainability information on top of dungeon visualization.
    
    Features:
    - Node hover tooltips showing grammar rules
    - Fitness contribution heatmap (red=bad, green=good)
    - Genealogy tree in side panel
    - Tile confidence overlay (opacity = confidence)
    """
    
    def __init__(
        self,
        explainability_manager: ExplainabilityManager,
        config: Optional[OverlayConfig] = None
    ):
        self.manager = explainability_manager
        self.config = config or OverlayConfig()
        
        # Cache for rendered elements
        self._tooltip_surface: Optional[pygame.Surface] = None
        self._heatmap_surface: Optional[pygame.Surface] = None
        self._genealogy_surface: Optional[pygame.Surface] = None
        
        # Interaction state
        self.hovered_node: Optional[int] = None
        self.selected_genome: Optional[str] = None
    
    def render(
        self,
        screen: pygame.Surface,
        mission_graph: nx.Graph,
        graph_layout: Dict[int, Tuple[int, int]],  # {node_id: (x, y)}
        mouse_pos: Tuple[int, int]
    ):
        """Render all debug overlays."""
        
        # 1. Update hover state
        self._update_hover_state(mouse_pos, graph_layout)
        
        # 2. Render fitness heatmap
        if self.config.show_fitness_heatmap and self.selected_genome:
            self._render_fitness_heatmap(screen, mission_graph, graph_layout)
        
        # 3. Render node tooltips
        if self.config.show_decision_tooltips and self.hovered_node is not None:
            self._render_node_tooltip(screen, mission_graph, mouse_pos)
        
        # 4. Render genealogy tree
        if self.config.show_genealogy_tree and self.selected_genome:
            self._render_genealogy_tree(screen)
        
        # 5. Render confidence overlay
        if self.config.show_confidence_overlay:
            self._render_confidence_overlay(screen)
    
    def _update_hover_state(self, mouse_pos: Tuple[int, int], graph_layout: Dict[int, Tuple[int, int]]):
        """Update which node is being hovered over."""
        self.hovered_node = None
        
        for node_id, (nx_x, nx_y) in graph_layout.items():
            dist = np.sqrt((mouse_pos[0] - nx_x)**2 + (mouse_pos[1] - nx_y)**2)
            if dist < 30:  # Hover radius
                self.hovered_node = node_id
                break
    
    def _render_node_tooltip(
        self,
        screen: pygame.Surface,
        mission_graph: nx.Graph,
        mouse_pos: Tuple[int, int]
    ):
        """
        Render tooltip showing why this node exists.
        
        Tooltip content:
        - Node ID and type
        - Grammar rules that created it
        - Fitness contribution
        - Confidence score
        """
        if self.hovered_node is None:
            return
        
        # Get decision traces for this node
        traces = self.manager.why_node(self.hovered_node, mission_graph)
        
        if not traces:
            return
        
        # Build tooltip text
        node_data = mission_graph.nodes[self.hovered_node]
        lines = [
            f"Node {self.hovered_node}: {node_data.get('type', 'UNKNOWN')}",
            "",
            "Created by:"
        ]
        
        for trace in traces:
            lines.append(f"  • {trace.description}")
            if trace.fitness_contribution is not None:
                sign = "+" if trace.fitness_contribution > 0 else ""
                lines.append(f"    Fitness: {sign}{trace.fitness_contribution:.3f}")
        
        # Render tooltip background
        font = pygame.font.Font(None, self.config.tooltip_font_size)
        tooltip_width = max(font.size(line)[0] for line in lines) + 20
        tooltip_height = len(lines) * (self.config.tooltip_font_size + 2) + 10
        
        tooltip_surf = pygame.Surface((tooltip_width, tooltip_height))
        tooltip_surf.fill((40, 40, 40))
        tooltip_surf.set_alpha(int(255 * self.config.overlay_alpha))
        
        # Render text
        y_offset = 5
        for line in lines:
            text_surf = font.render(line, True, (255, 255, 255))
            tooltip_surf.blit(text_surf, (10, y_offset))
            y_offset += self.config.tooltip_font_size + 2
        
        # Position tooltip near mouse
        tooltip_x = min(mouse_pos[0] + 20, screen.get_width() - tooltip_width)
        tooltip_y = min(mouse_pos[1] + 20, screen.get_height() - tooltip_height)
        
        screen.blit(tooltip_surf, (tooltip_x, tooltip_y))
    
    def _render_fitness_heatmap(
        self,
        screen: pygame.Surface,
        mission_graph: nx.Graph,
        graph_layout: Dict[int, Tuple[int, int]]
    ):
        """
        Render heatmap showing fitness contribution of each node.
        
        Colors:
        - Green: Positive fitness contribution
        - Red: Negative fitness contribution
        - Gray: Neutral
        """
        if not self.selected_genome:
            return
        
        # Get fitness attributions for selected genome
        generation = self.manager.genealogy[self.selected_genome].generation
        if generation not in self.manager.fitness_attributions:
            return
        
        if self.selected_genome not in self.manager.fitness_attributions[generation]:
            return
        
        attribution = self.manager.fitness_attributions[generation][self.selected_genome]
        
        # Draw colored circles around nodes based on fitness contribution
        for node_id in mission_graph.nodes():
            # Find traces affecting this node
            traces = self.manager.why_node(node_id, mission_graph)
            
            # Sum fitness contributions
            total_contribution = sum(
                t.fitness_contribution or 0.0
                for t in traces
                if t.fitness_contribution is not None
            )
            
            if total_contribution == 0:
                continue
            
            # Map contribution to color
            if total_contribution > 0:
                # Green for positive
                intensity = min(int(255 * abs(total_contribution) / 0.1), 255)
                color = (0, intensity, 0)
            else:
                # Red for negative
                intensity = min(int(255 * abs(total_contribution) / 0.1), 255)
                color = (intensity, 0, 0)
            
            # Draw halo around node
            nx_x, nx_y = graph_layout[node_id]
            pygame.draw.circle(screen, color, (int(nx_x), int(nx_y)), 40, 3)
    
    def _render_genealogy_tree(self, screen: pygame.Surface):
        """
        Render genealogy tree in side panel.
        
        Shows evolutionary lineage with:
        - Parent-child connections
        - Fitness of each genome
        - Mutations applied
        """
        if not self.selected_genome:
            return
        
        # Get lineage
        lineage = self.manager.get_genealogy_lineage(self.selected_genome)
        
        if not lineage:
            return
        
        # Create side panel
        panel_width = 300
        panel_height = screen.get_height()
        panel_x = screen.get_width() - panel_width
        
        panel_surf = pygame.Surface((panel_width, panel_height))
        panel_surf.fill((30, 30, 30))
        panel_surf.set_alpha(int(255 * self.config.overlay_alpha))
        
        # Render title
        font = pygame.font.Font(None, 24)
        title_surf = font.render("Genealogy", True, (255, 255, 255))
        panel_surf.blit(title_surf, (10, 10))
        
        # Render lineage
        y_offset = 50
        font_small = pygame.font.Font(None, 16)
        
        for i, gen_record in enumerate(lineage):
            # Indent based on generation
            indent = i * 20
            
            # Render genome info
            text = f"Gen {gen_record.generation}: {gen_record.genome_id[:12]}..."
            text_surf = font_small.render(text, True, (200, 200, 200))
            panel_surf.blit(text_surf, (10 + indent, y_offset))
            y_offset += 20
            
            # Render fitness
            fitness_text = f"  Fitness: {gen_record.fitness:.3f}"
            fitness_surf = font_small.render(fitness_text, True, (150, 255, 150))
            panel_surf.blit(fitness_surf, (10 + indent, y_offset))
            y_offset += 20
            
            # Render mutations
            if gen_record.mutations_applied:
                mut_text = f"  Mutations: {len(gen_record.mutations_applied)}"
                mut_surf = font_small.render(mut_text, True, (150, 150, 255))
                panel_surf.blit(mut_surf, (10 + indent, y_offset))
                y_offset += 20
            
            # Draw connection line to next
            if i < len(lineage) - 1:
                pygame.draw.line(
                    panel_surf,
                    (100, 100, 100),
                    (15 + indent, y_offset),
                    (35 + indent, y_offset + 10),
                    2
                )
                y_offset += 10
            
            y_offset += 10  # Spacing
        
        screen.blit(panel_surf, (panel_x, 0))
    
    def _render_confidence_overlay(self, screen: pygame.Surface):
        """
        Render confidence overlay on tiles.
        
        Tiles with low confidence are highlighted in red.
        """
        # This would render tile-level confidence from LogicNet
        # Implementation depends on tile rendering system
        pass
    
    def handle_click(self, mouse_pos: Tuple[int, int], graph_layout: Dict[int, Tuple[int, int]]):
        """Handle click event to select genome."""
        # Check if clicked on a node
        for node_id, (nx_x, nx_y) in graph_layout.items():
            dist = np.sqrt((mouse_pos[0] - nx_x)**2 + (mouse_pos[1] - nx_y)**2)
            if dist < 30:
                # Find genome that created this node
                traces = self.manager.why_node(node_id, None)
                if traces:
                    genome_id = traces[0].metadata.get('genome_id')
                    if genome_id:
                        self.selected_genome = genome_id
                        print(f"Selected genome: {genome_id}")
                break


# ============================================================================
# INTEGRATION WITH GUI_RUNNER.PY
# ============================================================================

"""
# In gui_runner.py, add explainability overlay:

from src.utils.explainability import ExplainabilityManager
from src.utils.explainability_gui import ExplainabilityDebugOverlay, OverlayConfig

class DungeonGeneratorGUI:
    def __init__(self):
        # ... existing init ...
        
        # NEW: Explainability system
        self.explainability_manager = ExplainabilityManager(
            log_dir="outputs/explainability"
        )
        
        self.debug_overlay = ExplainabilityDebugOverlay(
            explainability_manager=self.explainability_manager,
            config=OverlayConfig(
                show_decision_tooltips=True,
                show_fitness_heatmap=True,
                show_genealogy_tree=False  # Toggle with F3
            )
        )
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                # NEW: Handle debug overlay interactions
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.debug_overlay.handle_click(event.pos, self.graph_layout)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F3:
                        # Toggle genealogy tree
                        self.debug_overlay.config.show_genealogy_tree = not self.debug_overlay.config.show_genealogy_tree
            
            # Render dungeon
            self.render_dungeon()
            
            # NEW: Render debug overlay on top
            if self.show_debug_overlay:
                self.debug_overlay.render(
                    screen=self.screen,
                    mission_graph=self.current_mission_graph,
                    graph_layout=self.graph_layout,
                    mouse_pos=pygame.mouse.get_pos()
                )
            
            pygame.display.flip()


# Pass explainability manager to pipeline components:

class NeuralSymbolicDungeonPipeline:
    def __init__(self, ..., explainability_manager: Optional[ExplainabilityManager] = None):
        self.explainability = explainability_manager
    
    def generate_dungeon(self, ...):
        # Trace decisions as they happen
        if self.explainability:
            # In EvolutionaryDirector:
            from src.utils.explainability import trace_grammar_rule_application
            
            for rule_idx, rule in enumerate(genome):
                nodes_before = set(graph.nodes())
                edges_before = set(graph.edges())
                
                # Apply rule
                graph = rule.apply(graph)
                
                # Trace what changed
                nodes_added = list(set(graph.nodes()) - nodes_before)
                edges_added = list(set(graph.edges()) - edges_before)
                
                trace_grammar_rule_application(
                    explainability_manager=self.explainability,
                    rule_name=rule.__class__.__name__,
                    genome_id=genome_id,
                    nodes_added=nodes_added,
                    edges_added=edges_added,
                    execution_order=rule_idx
                )
"""


# ============================================================================
# COMMAND-LINE EXPLAINABILITY TOOL
# ============================================================================

def cli_explain_dungeon(trace_file: str):
    """
    Command-line tool to query explainability traces.
    
    Usage:
        python -m src.utils.explainability_gui outputs/explainability/traces.json
    """
    import sys
    
    manager = ExplainabilityManager()
    manager.load_json(trace_file)
    
    print("=" * 60)
    print("KLTN Explainability CLI")
    print("=" * 60)
    print(f"Loaded {len(manager.traces)} decision traces")
    print(f"Genomes in genealogy: {len(manager.genealogy)}")
    print()
    
    while True:
        print("\nCommands:")
        print("  1. Why node <node_id>")
        print("  2. Why edge <source> <target>")
        print("  3. Fitness attribution <genome_id>")
        print("  4. Genealogy <genome_id>")
        print("  5. Export HTML report")
        print("  6. Quit")
        
        choice = input("\nEnter command number: ").strip()
        
        if choice == "1":
            node_id = int(input("Node ID: "))
            traces = manager.why_node(node_id, None)
            print(f"\n=== Why Node {node_id} Exists ===")
            for trace in traces:
                print(f"  • {trace.description}")
                if trace.fitness_contribution:
                    print(f"    Fitness Δ: {trace.fitness_contribution:+.3f}")
        
        elif choice == "2":
            source = int(input("Source node: "))
            target = int(input("Target node: "))
            traces = manager.why_edge(source, target)
            print(f"\n=== Why Edge ({source}, {target}) Exists ===")
            for trace in traces:
                print(f"  • {trace.description}")
        
        elif choice == "3":
            genome_id = input("Genome ID: ").strip()
            generation = int(input("Generation: "))
            contributions = manager.what_caused_fitness_change(genome_id, generation)
            print(f"\n=== Fitness Attribution for {genome_id} ===")
            print(f"Top {len(contributions)} contributors:")
            for decision_id, delta in contributions:
                print(f"  {delta:+.3f} - {decision_id}")
        
        elif choice == "4":
            genome_id = input("Genome ID: ").strip()
            lineage = manager.get_genealogy_lineage(genome_id)
            print(f"\n=== Genealogy for {genome_id} ===")
            for i, gen_record in enumerate(lineage):
                indent = "  " * i
                print(f"{indent}Gen {gen_record.generation}: {gen_record.genome_id}")
                print(f"{indent}  Fitness: {gen_record.fitness:.3f}")
                print(f"{indent}  Mutations: {len(gen_record.mutations_applied)}")
        
        elif choice == "5":
            output = input("Output path (default: report.html): ").strip() or "report.html"
            manager.generate_html_report(output)
            print(f"✓ Report saved to {output}")
        
        elif choice == "6":
            break
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli_explain_dungeon(sys.argv[1])
    else:
        print("Usage: python explainability_gui.py <trace_file.json>")
