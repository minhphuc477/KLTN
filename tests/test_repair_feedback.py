import torch

from src.pipeline.repair_feedback import build_neighbor_boundary_inpaint_inputs


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
