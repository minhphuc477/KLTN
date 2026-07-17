import torch
from torch import nn

from src.core.condition_encoder import create_condition_encoder


class _FixedLocalEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, neighbor_latents, boundary_constraints, position):
        del neighbor_latents, position
        return torch.ones(
            int(boundary_constraints.shape[0]),
            self.output_dim,
            device=boundary_constraints.device,
            dtype=boundary_constraints.dtype,
        )


class _IdentityGlobalEncoder(nn.Module):
    def forward(self, node_features, edge_index, **kwargs):
        del edge_index, kwargs
        return node_features


class _RecordingFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local = None
        self.global_tokens = None
        self.mask = None

    def forward(self, c_local, c_global, mask=None):
        self.local = c_local.detach().clone()
        self.global_tokens = c_global.detach().clone()
        self.mask = None if mask is None else mask.detach().clone()
        return c_local


class _SelectFusedOutput(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, combined):
        return combined[:, : self.output_dim]


def test_current_node_anchors_query_without_collapsing_graph_attention():
    output_dim = 8
    encoder = create_condition_encoder(
        latent_dim=4,
        node_feature_dim=output_dim,
        hidden_dim=16,
        output_dim=output_dim,
        num_attention_heads=2,
        dropout=0.0,
    )
    recorder = _RecordingFusion()
    encoder.local_encoder = _FixedLocalEncoder(output_dim)
    encoder.global_encoder = _IdentityGlobalEncoder()
    encoder.fusion = recorder
    encoder.output_proj = _SelectFusedOutput(output_dim)

    graph_tokens = torch.arange(3 * output_dim, dtype=torch.float32).reshape(3, output_dim)
    node_mask = torch.tensor([True, True, False])
    condition, returned_tokens = encoder(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        boundary_constraints=torch.zeros(1, 8),
        position=torch.zeros(1, 2),
        node_features=graph_tokens,
        edge_index=torch.empty(2, 0, dtype=torch.long),
        node_mask=node_mask,
        current_node_idx=1,
        return_global_tokens=True,
    )

    assert recorder.global_tokens is not None
    assert tuple(recorder.global_tokens.shape) == (1, 3, output_dim)
    assert torch.equal(recorder.global_tokens, graph_tokens.unsqueeze(0))
    assert torch.equal(recorder.mask, node_mask.unsqueeze(0))
    assert torch.equal(recorder.local, torch.ones(1, output_dim) + graph_tokens[1:2])
    assert torch.equal(condition, recorder.local)
    assert torch.equal(returned_tokens, graph_tokens.unsqueeze(0))


def test_current_node_cannot_select_a_masked_graph_token():
    encoder = create_condition_encoder(
        latent_dim=4,
        node_feature_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_attention_heads=2,
        dropout=0.0,
    )
    encoder.local_encoder = _FixedLocalEncoder(8)
    encoder.global_encoder = _IdentityGlobalEncoder()

    try:
        encoder(
            neighbor_latents={"N": None, "S": None, "E": None, "W": None},
            boundary_constraints=torch.zeros(1, 8),
            position=torch.zeros(1, 2),
            node_features=torch.zeros(2, 8),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            node_mask=torch.tensor([True, False]),
            current_node_idx=1,
        )
    except ValueError as exc:
        assert "masked graph token" in str(exc)
    else:
        raise AssertionError("A masked current node must violate the conditioning contract")
