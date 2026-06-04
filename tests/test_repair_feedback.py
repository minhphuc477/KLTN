import numpy as np
import torch

from src.pipeline.repair_feedback import (
    build_latent_edit_mask,
    build_neighbor_boundary_inpaint_inputs,
    logicnet_guided_inpaint_room,
    wfc_guided_inpaint_room,
)


def test_build_neighbor_boundary_inpaint_inputs_preserves_edges():
    base = torch.zeros(1, 2, 4, 4)
    north = torch.full((1, 2, 4, 4), 10.0)
    south = torch.full((1, 2, 4, 4), 20.0)
    west = torch.full((1, 2, 4, 4), 30.0)
    east = torch.full((1, 2, 4, 4), 40.0)

    ref, mask, has_constraints = build_neighbor_boundary_inpaint_inputs(
        base_latent=base,
        neighbor_latents={"N": north, "S": south, "W": west, "E": east},
        band=1,
    )

    assert has_constraints is True
    assert ref.shape == base.shape
    assert mask.shape == (1, 1, 4, 4)

    # Check directional copy behavior.
    # Corners are shared by two directions; assert non-corner segments.
    assert torch.all(ref[:, :, 0, 1:-1] == 10.0)  # top from north bottom band
    assert torch.all(ref[:, :, -1, 1:-1] == 20.0)  # bottom from south top band
    assert torch.all(ref[:, :, 1:-1, 0] == 30.0)  # left from west right band
    assert torch.all(ref[:, :, 1:-1, -1] == 40.0)  # right from east left band

    # Preserved boundary mask should be zero on all borders.
    assert torch.all(mask[:, :, 0, :] == 0.0)
    assert torch.all(mask[:, :, -1, :] == 0.0)
    assert torch.all(mask[:, :, :, 0] == 0.0)
    assert torch.all(mask[:, :, :, -1] == 0.0)


def test_build_neighbor_boundary_inpaint_inputs_handles_missing_neighbors():
    base = torch.randn(1, 2, 4, 4)
    ref, mask, has_constraints = build_neighbor_boundary_inpaint_inputs(
        base_latent=base,
        neighbor_latents={"N": None, "S": None, "W": None, "E": None},
        band=1,
    )

    assert has_constraints is False
    assert torch.allclose(ref, base)
    assert torch.all(mask == 1.0)


def test_build_neighbor_boundary_inpaint_inputs_aligns_batch_and_spatial_shape():
    base = torch.zeros(2, 2, 4, 4)
    # Batch-1 + different spatial size should be aligned and broadcast.
    north = torch.full((1, 2, 6, 6), 7.0)

    ref, mask, has_constraints = build_neighbor_boundary_inpaint_inputs(
        base_latent=base,
        neighbor_latents={"N": north, "S": None, "W": None, "E": None},
        band=1,
    )

    assert has_constraints is True
    assert ref.shape == base.shape
    assert mask.shape == (2, 1, 4, 4)
    assert torch.all(ref[:, :, 0, :] == 7.0)
    assert torch.all(mask[:, :, 0, :] == 0.0)


def test_build_latent_edit_mask_downsamples_dead_end_region():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True

    latent_mask = build_latent_edit_mask(
        mask,
        latent_h=2,
        latent_w=2,
        device=torch.device("cpu"),
    )

    assert latent_mask.shape == (1, 1, 2, 2)
    assert latent_mask[0, 0, 0, 0].item() == 1.0
    assert latent_mask[0, 0, 1, 1].item() == 0.0


def test_logicnet_guided_inpaint_room_restores_guidance_scale_and_aliases_old_name():
    class _FakeVQVAE:
        num_classes = 3

        def encode(self, x_0):
            return torch.zeros(1, 2, 2, 2, dtype=x_0.dtype), None

        def decode(self, z):
            logits = torch.zeros(1, 3, 4, 4, dtype=z.dtype)
            logits[:, 2] = 5.0
            return logits

    class _Guidance:
        def __init__(self):
            self.guidance_scale = 1.5

    class _FakeDiffusion:
        def __init__(self):
            self.guidance = _Guidance()
            self.seen = None

        def inpaint(self, **kwargs):
            self.seen = kwargs
            assert self.guidance.guidance_scale == 3.0
            return kwargs["x_0"] + 1.0

    diffusion = _FakeDiffusion()
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[0, 0] = 1
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    out = logicnet_guided_inpaint_room(
        current_grid=grid,
        dead_end_mask=mask,
        condition=torch.zeros(1, 8),
        graph_data={},
        num_diffusion_steps=4,
        seed=1,
        device=torch.device("cpu"),
        vqvae=_FakeVQVAE(),
        diffusion=diffusion,
        num_classes=3,
        noise_strength=0.25,
        guidance_scale_multiplier=2.0,
    )

    assert out[0, 0] == 2
    assert out[1, 1] == 0
    assert diffusion.guidance.guidance_scale == 1.5
    assert diffusion.seen["num_steps"] == 8
    assert diffusion.seen["noise_strength"] == 0.25

    alias_out = wfc_guided_inpaint_room(
        current_grid=grid,
        dead_end_mask=np.zeros((4, 4), dtype=bool),
        condition=torch.zeros(1, 8),
        graph_data={},
        num_diffusion_steps=4,
        seed=None,
        device=torch.device("cpu"),
        vqvae=_FakeVQVAE(),
        diffusion=diffusion,
        num_classes=3,
    )
    assert np.array_equal(alias_out, grid)
