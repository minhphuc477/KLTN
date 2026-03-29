# Discrete Masked Diffusion Fit for KLTN

Date: 2026-03-29

## Conclusion

Discrete masked room generation is a good fit for this codebase, but it should
exist as a parallel room-generation path rather than an immediate replacement
for the current VQ-VAE + latent diffusion stack.

## Why it fits

- Zelda rooms are small: `ROOM_HEIGHT x ROOM_WIDTH = 16 x 11`, only 176 tokens.
- Hard constraints such as doors, start, and goal are much easier to enforce in
  token space than in continuous latent space.
- The current graph-conditioned pipeline already provides the right inputs:
  graph tokens, node positions, TPE, room topology priors, and room-local role
  semantics.
- MaskGIT-style iterative unmasking naturally supports partial canvases and
  fixed known tokens.

## Why it should not fully replace the latent path yet

- The current condition encoder still has latent-oriented behavior, especially
  around neighbor context and legacy boundary handling.
- The pipeline has existing repair, inpainting, and evaluation flows built
  around the latent model.
- The repo already has trained and training-capable latent components, while
  the discrete path is new and needs its own checkpoints.

## Practical architecture decision

Implement a second engine for Block IV:

- `room_generator_mode="latent_diffusion"` keeps the current path.
- `room_generator_mode="discrete_masked"` uses a graph-conditioned masked-token
  model over full room grids.

Keep these shared:

- Block I topology generation
- Block III graph/local conditioning
- room topology priors
- symbolic repair and stitching

## Implemented direction

The discrete path added in this repo uses:

- token embedding over the full `16 x 11` room grid
- the existing U-Net denoiser as the spatial backbone
- graph-conditioned cross-attention and room topology maps
- MaskGIT-style iterative confidence-based unmasking
- hard-fixed door/start/goal tokens at sampling time

Training is pure masked-token prediction with cross-entropy on masked tiles.

## Recommended next steps

1. Train masked-room checkpoints on the room-level Zelda corpus.
2. Compare door validity and repair load against the latent baseline.
3. If the discrete path consistently lowers repair pressure, expand it to
   replace more latent-specific boundary code.

## References

- MaskGIT: Chang et al., CVPR 2022
- D3PM: Austin et al., NeurIPS 2021
