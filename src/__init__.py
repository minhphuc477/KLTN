"""
KLTN Source Package - G-LMDM Canonical Architecture & H-MOLQD Comparators
=========================================================================

Graph-Conditioned Logic-Regularized Masked Discrete Model (G-LMDM) for
procedural Legend of Zelda dungeon generation with verified neuro-symbolic constraints.

Submodules:
- core: Neural components (GraphGPS ConditionEncoder, DiscreteMaskedRoomModel,
        Soft Bellman-Ford LogicNet, SymbolicRefiner) + H-MOLQD comparators
- data_processing: Intelligent data adapter and VGLC visual extraction
- generation: Mission Grammar (Block I) and constructive PuzzleGenerator (Block VI)
- pipeline: Room processing, dungeon assembler, and verification orchestration
- evaluation: ExternalValidator, StateSpaceAStar, P-CBS, and MAP-Elites
- simulation: Tile-state progression oracle and cognitive bounded search
- visualization: GUI and rendering

Canonical 5-Block G-LMDM Runtime Pipeline:
    Block 0:   IntelligentDataAdapter - Corpus parsing and V4/D4 alignment
    Block I:   MissionGrammar - Key-lock/switch mission DAG generation
    Block III: DualStreamConditionEncoder - Causal GraphGPS + RRWP conditioning
    Block IV:  DiscreteMaskedRoomModel - Absorbing-state 16x11 masked discrete denoiser (84.8K params)
    Block V:   LogicNet - Direct 16x11 Soft Bellman-Ford regularization (alpha_logic=0.1)
    Block VI:  PuzzleGenerator + SymbolicRefiner - Stateful puzzle & minimum-edit repair
    Block VII: StateSpaceAStar Oracle + P-CBS Proxy - Formal & bounded-agent evaluation

Continuous Comparator Branch (H-MOLQD, retained strictly for RQ1/H1 ablations):
    Block II:  SemanticVQVAE - Continuous latent tokenizer
    Block IVb: LatentDiffusionModel - UNet[1,2] / DiT+Flow continuous latent comparators
"""

__version__ = "1.0.0"

__all__ = ['core', 'data_processing', 'evaluation', 'generation', 'pipeline', 'simulation', 'visualization']
