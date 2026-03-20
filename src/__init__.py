"""
KLTN Source Package - H-MOLQD Architecture
===========================================

Hybrid Masked-diffusion Optimization with Logical Quality Diversity
for procedural Zelda dungeon generation with neuro-symbolic constraints.

Submodules:
- core: Neural components (VQ-VAE, Diffusion, LogicNet, Refiner) + definitions
- data_processing: Intelligent data adapter and visual extraction
- evaluation: Validator and MAP-Elites quality diversity
- simulation: Validation and simulation 
- visualization: GUI and rendering

Architecture Blocks (canonical runtime pipeline):
    Block 0:   IntelligentDataAdapter - Corpus parsing and alignment
    Block I:   EvolutionaryTopologyGenerator - Mission graph generation
    Block II:  SemanticVQVAE - Discrete representation learning
    Block III: DualStreamConditionEncoder - Contextual awareness
    Block IV:  LatentDiffusionModel - Global layout generation
    Block V:   LogicNet - Differentiable solvability approximation
    Block VI:  SymbolicRefiner - Path-guided constraint repair
    Block VII: MAP-Elites + ExternalValidator - Quality-diversity evaluation
"""

__version__ = "1.0.0"

__all__ = ['core', 'data_processing', 'evaluation', 'simulation', 'visualization']
