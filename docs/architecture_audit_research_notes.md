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

## Implementation Implications

- The U-Net attention residual now lives inside the Pre-LN self-attention
  block, avoiding a double residual around attention.
- LogicNet temperature annealing remains explicit and is called through
  `anneal_temperature`, with `update_temperature` retained as a compatibility
  alias.
- Fast-sampler configuration defaults do not claim LCM-LoRA acceleration unless
  a real distilled backend is provided. The implemented backend is repo-specific
  `consistency_lora` from `src/train_lcm.py`, metadata-gated through the
  underlying graph-aware latent diffusion runtime; arbitrary Stable Diffusion
  LCM-LoRA checkpoints remain incompatible with this custom latent space.
- Tile-pattern distribution metrics were added as a discrete corpus comparison
  primitive; they are intended to supplement, not replace, human playtests and
  solver-based validation.
