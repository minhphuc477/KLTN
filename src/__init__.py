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

Architecture Blocks:
    Block I:   IntelligentDataAdapter - Data ingestion & topological alignment
    Block II:  SemanticVQVAE - Discrete representation learning
    Block III: DualStreamConditionEncoder - Contextual awareness
    Block IV:  LatentDiffusionModel - Global layout generation
    Block V:   LogicNet - Differentiable solvability approximation
    Block VI:  MAPElites + ExternalValidator - Scientific evaluation
    Block VII: SymbolicRefiner - Path-guided constraint repair
"""

__version__ = "1.0.0"
__author__ = "KLTN Thesis Project"

__all__ = ['core', 'data_processing', 'evaluation', 'simulation', 'visualization']
