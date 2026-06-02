# Architecture Audit Research Notes

These notes record the external references used while addressing the H-MOLQD
architecture audit. They are not an experimental-results claim; they document
why the implemented fixes are defensible and where empirical validation is
still required.

## References

- Pre-LN residual placement and Transformer training stability: Xiong et al.,
  "On Layer Normalization in the Transformer Architecture",
  https://arxiv.org/abs/2002.04745.
- Temperature annealing for categorical relaxations: Jang et al.,
  "Categorical Reparameterization with Gumbel-Softmax",
  https://arxiv.org/abs/1611.01144.
- Discrete level-corpus baselines and tile-pattern comparison context:
  Summerville et al., "The VGLC: The Video Game Level Corpus",
  https://arxiv.org/abs/1606.07487.
- Few-step latent diffusion acceleration requires a trained/distilled
  consistency model or LoRA: Luo et al., "Latent Consistency Models",
  https://arxiv.org/abs/2310.04378, and "LCM-LoRA",
  https://arxiv.org/abs/2311.05556. Song et al. provide the broader
  consistency-model basis at https://arxiv.org/abs/2303.01469.
- The authors' full-model LCM distillation script separates a frozen diffusion
  teacher, online student, and EMA target student:
  https://github.com/luosiallen/latent-consistency-model/blob/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_sd_wds.py.
  Their LoRA example still uses adjacent trajectory points but omits the
  separate EMA target:
  https://github.com/luosiallen/latent-consistency-model/blob/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_lora_sd_wds.py.
- PyTorch documents epoch-level cosine learning-rate annealing through
  `CosineAnnealingLR`:
  https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html.
- WFC should preserve recursively propagated constraint supports, not only
  one-hop neighbor updates: Karth and Smith, "WaveFunctionCollapse is
  Constraint Solving in the Wild",
  https://www.pcgworkshop.com/archive/karth2017wavefunctioncollapse.pdf.
- MAP-Elites archives elites by user-selected dimensions of phenotypic
  variation: Mouret and Clune, "Illuminating Search Spaces by Mapping Elites",
  https://arxiv.org/abs/1504.04909.
- Pydantic v2 models validate input through `model_validate()` and can forbid
  extra fields through model configuration:
  https://docs.pydantic.dev/latest/concepts/models/.
- `pytest-timeout` supports a global configured timeout:
  https://pypi.org/project/pytest-timeout/.
- NumPy documents that `convolve(..., mode="same")` still exposes boundary
  effects and that `pad(..., mode="edge")` replicates array edge values:
  https://numpy.org/doc/stable/reference/generated/numpy.convolve and
  https://numpy.org/doc/stable/reference/generated/numpy.pad.html.
- NetworkX defines undirected degree as the number of adjacent edges and
  directed out-degree as the number of outgoing edges. Branching metrics must
  distinguish corridor adjacency from forward choices:
  https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.degree.html
  and
  https://networkx.org/documentation/stable/_modules/networkx/classes/digraph.html.
- The University of Alberta Sokoban rules specify one-at-a-time stone pushes.
  Dynamic block occupancy therefore has to be resolved from search state, not
  only from the immutable source grid:
  https://webdocs.cs.ualberta.ca/~games/Sokoban/thegame.html.

## Implementation Implications

- The U-Net attention residual now lives inside the Pre-LN self-attention
  block, avoiding a double residual around attention.
- LogicNet temperature annealing remains explicit and is called through
  `anneal_temperature`, with `update_temperature` retained as a compatibility
  alias.
- Fast-sampler configuration defaults do not claim LCM-LoRA acceleration unless
  a real distilled backend is provided. The implemented backend is repo-specific
  `consistency_lora` from `src/train_lcm.py`: a frozen graph-aware diffusion
  teacher advances adjacent DDIM trajectory points, an online LoRA student
  learns the high-noise consistency output, and an EMA target student provides
  the lower-noise target. Resume checkpoints retain both adapters and the
  deployable adapter export uses the EMA target. It remains metadata-gated
  through the underlying graph-aware latent diffusion runtime; arbitrary Stable
  Diffusion LCM-LoRA checkpoints remain incompatible with this custom latent
  space.
- Gaussian-VAE training now follows the existing VQ-VAE policy with resumable
  epoch-level `CosineAnnealingLR` state.
- Tile-pattern distribution metrics were added as a discrete corpus comparison
  primitive; they are intended to supplement, not replace, human playtests and
  solver-based validation.
- Weighted WFC now treats zero-support cells as contradictions and recursively
  propagates support reductions instead of restoring an unconstrained prior.
- Topology generation records the pre/post repair fitness and feasibility
  shift so exported phenotypes can be analyzed separately from pre-repair
  individuals.
- Pacing smoothing edge-pads the normalized tension curve before applying the
  three-tap convolution, preserving first-room and final-boss tension.
- Frustration scoring uses excess decision branching rather than raw average
  degree and uses room-level goal density rather than an all-or-nothing goal
  flag. A single boss room no longer erases branching confusion.
- Flat and underspecified dungeon inputs no longer receive positive flow or
  pacing progression credit.
- Push-block search resolves current block destinations before static vacated
  origins. Sequential validation, parallel A*, and P-CBS now share dynamic
  occupancy semantics.
- Advanced pipeline fun evaluation follows a resolved start-to-goal graph path,
  receives the NetworkX graph expected by the evaluator, and retains graph and
  entity semantics for bosses, goals, puzzles, locks, rewards, and recovery.
