import logging
from types import SimpleNamespace
import networkx as nx
import numpy as np
import pytest
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_TOPOLOGY_CHANNELS, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.discrete_masked_model import DiscreteMaskedRoomModel, create_discrete_masked_model
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.pipeline.room_topology_conditioning import build_room_semantic_anchor_points
from src.train_masked_room import (
    MaskedRoomTrainer,
    MaskedRoomTrainingConfig,
    _create_masked_room_dataloaders,
    train_masked_room,
)
from src.zelda_data.zelda_loader import DungeonBatchSampler


class _DatasetLoaderStub:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


class _DummyMaskedConditionEncoder:
    def __init__(self, output_dim: int = 8):
        self.output_dim = output_dim
        self.captured_reference_room_maps = None
        self.captured_style_id = None

    def __call__(
        self,
        *,
        neighbor_latents,
        boundary_constraints,
        position,
        node_features,
        edge_index,
        edge_features=None,
        edge_rrwp=None,
        tpe=None,
        current_node_distance=None,
        node_mask=None,
        current_node_idx=None,
        reference_room_maps=None,
        style_id=None,
        return_global_tokens=False,
    ):
        _ = (
            neighbor_latents,
            boundary_constraints,
            position,
            node_features,
            edge_index,
            edge_features,
            edge_rrwp,
            tpe,
            current_node_distance,
            node_mask,
            current_node_idx,
            style_id,
        )
        self.captured_reference_room_maps = reference_room_maps
        self.captured_style_id = style_id
        room_anchor = torch.full((1, self.output_dim), 7.0, dtype=torch.float32)
        if return_global_tokens:
            global_tokens = torch.full((1, int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)
            return room_anchor, global_tokens
        return room_anchor

    def encode_global_only(self, *args, **kwargs):
        node_features = args[0]
        _ = kwargs
        return torch.full((int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)

    def encode_local_only(self, *args, **kwargs):
        _ = (args, kwargs)
        return torch.full((1, self.output_dim), 5.0, dtype=torch.float32)


def test_discrete_masked_model_respects_fixed_tokens():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=64,
        context_dim=256,
        num_steps=3,
    )
    context = torch.zeros(1, 1, 256)
    fixed_tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    fixed_mask = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.bool)
    fixed_tokens[0, 0, 5] = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    fixed_tokens[0, 8, 5] = int(SEMANTIC_PALETTE["START"])
    fixed_mask[0, 0, 5] = True
    fixed_mask[0, 8, 5] = True

    tokens, _logits, _hidden = model.sample(
        context=context,
        fixed_tokens=fixed_tokens,
        fixed_mask=fixed_mask,
        num_steps=2,
        seed=123,
    )

    assert int(tokens[0, 0, 5]) == int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    assert int(tokens[0, 8, 5]) == int(SEMANTIC_PALETTE["START"])


def test_discrete_masked_model_reports_iterative_refinement_metrics():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=32,
        num_steps=3,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
    )
    context = torch.zeros(1, 1, 32)
    tokens, logits, hidden, metrics = model.sample(
        context=context,
        num_steps=3,
        stochastic=False,
        corrector_steps=1,
        corrector_mask_ratio=0.1,
        seed=7,
        return_sampling_metrics=True,
    )

    assert tuple(tokens.shape) == (1, ROOM_HEIGHT, ROOM_WIDTH)
    assert tuple(logits.shape[-2:]) == (ROOM_HEIGHT, ROOM_WIDTH)
    assert hidden.shape[0] == 1
    assert metrics["masked_refinement_steps_requested"] == pytest.approx(3.0)
    assert metrics["masked_refinement_steps_executed"] == pytest.approx(3.0)
    assert metrics["masked_corrector_rounds_executed"] == pytest.approx(1.0)
    assert metrics["masked_initial_editable_tokens"] == pytest.approx(float(ROOM_HEIGHT * ROOM_WIDTH))
    assert metrics["masked_final_unresolved_tokens"] == pytest.approx(0.0)


def test_discrete_masked_training_step_embedding_tracks_mask_ratio():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=32,
        num_steps=5,
    )
    target = torch.randint(0, 44, (1, ROOM_HEIGHT, ROOM_WIDTH))
    context = torch.zeros(1, 1, 32)

    _loss_high, metrics_high, aux_high = model.training_loss(
        target,
        context,
        min_mask_ratio=1.0,
        max_mask_ratio=1.0,
        return_aux=True,
    )
    _loss_low, metrics_low, aux_low = model.training_loss(
        target,
        context,
        min_mask_ratio=0.0,
        max_mask_ratio=0.0,
        return_aux=True,
    )

    assert int(aux_high["step"].item()) == 4
    assert metrics_high["step_mean"] == pytest.approx(4.0)
    assert int(aux_low["step"].item()) == 0
    assert metrics_low["step_mean"] == pytest.approx(0.0)


def test_masked_backbone_default_uses_original_concat_encoder(monkeypatch):
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=16,
        context_dim=8,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    step = torch.zeros(1, dtype=torch.long)
    context = torch.randn(1, 37, 8)
    seen = {}

    def fake_encoder(sequence, *args, **kwargs):
        _ = (args, kwargs)
        seen["encoder_seq_len"] = int(sequence.shape[1])
        return sequence

    def forbidden_decoder(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("cross-decoder ablation should not run by default")

    monkeypatch.setattr(model.backbone.encoder, "forward", fake_encoder)
    monkeypatch.setattr(model.backbone.decoder, "forward", forbidden_decoder)

    logits = model(tokens, step, context)

    assert logits.shape == (1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    assert seen["encoder_seq_len"] == ROOM_HEIGHT * ROOM_WIDTH + 37


def test_masked_backbone_cross_decoder_ablation_uses_room_tokens_only_for_self_attention(monkeypatch):
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=16,
        context_dim=8,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
        unet_dropout=0.0,
        context_attention_mode="cross_decoder",
    )
    tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    step = torch.zeros(1, dtype=torch.long)
    context = torch.randn(1, 37, 8)
    seen = {}

    def fake_encoder(room_tokens, *args, **kwargs):
        _ = (args, kwargs)
        seen["room_seq_len"] = int(room_tokens.shape[1])
        return room_tokens

    def fake_decoder(tgt, memory, *args, **kwargs):
        seen["decoder_tgt_len"] = int(tgt.shape[1])
        seen["decoder_memory_len"] = int(memory.shape[1])
        seen["memory_key_padding_mask"] = kwargs.get("memory_key_padding_mask")
        return tgt

    monkeypatch.setattr(model.backbone.encoder, "forward", fake_encoder)
    monkeypatch.setattr(model.backbone.decoder, "forward", fake_decoder)

    logits = model(tokens, step, context)

    assert logits.shape == (1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    assert seen["room_seq_len"] == ROOM_HEIGHT * ROOM_WIDTH
    assert seen["decoder_tgt_len"] == ROOM_HEIGHT * ROOM_WIDTH
    assert seen["decoder_memory_len"] == 37


def test_masked_concat_encoder_respects_context_node_mask():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = create_discrete_masked_model(
            num_classes=44,
            hidden_dim=16,
            context_dim=8,
            num_steps=2,
            unet_channel_mult=(1,),
            unet_num_res_blocks=1,
            unet_num_heads=4,
            unet_dropout=0.0,
            context_attention_mode="concat_encoder",
        )
        model.eval()
        tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
        step = torch.zeros(1, dtype=torch.long)
        context = torch.randn(1, 5, 8)
        graph_data = {"node_mask": torch.tensor([[True, True, False, False, False]])}

        baseline = model(tokens, step, context, graph_data=graph_data)
        changed_masked = context.clone()
        changed_masked[:, 2:] = changed_masked[:, 2:] * 250.0 - 13.0
        masked_same = model(tokens, step, changed_masked, graph_data=graph_data)
        changed_valid = context.clone()
        changed_valid[:, :2] = changed_valid[:, :2] + 1.0
        valid_changed = model(tokens, step, changed_valid, graph_data=graph_data)

        assert torch.allclose(baseline, masked_same, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(baseline, valid_changed)


def test_masked_backbone_ignores_masked_context_tokens():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = create_discrete_masked_model(
            num_classes=44,
            hidden_dim=16,
            context_dim=8,
            num_steps=2,
            unet_channel_mult=(1,),
            unet_num_res_blocks=1,
            unet_num_heads=4,
            unet_dropout=0.0,
            context_attention_mode="cross_decoder",
        )
        model.eval()
        tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
        step = torch.zeros(1, dtype=torch.long)
        context = torch.randn(1, 4, 8)
        graph_data = {"node_mask": torch.tensor([[True, True, False, False]])}

        baseline = model(tokens, step, context, graph_data=graph_data)
        changed_masked = context.clone()
        changed_masked[:, 2:] = changed_masked[:, 2:] * 100.0 + 17.0
        masked_same = model(tokens, step, changed_masked, graph_data=graph_data)
        changed_valid = context.clone()
        changed_valid[:, :2] = changed_valid[:, :2] + 1.0
        valid_changed = model(tokens, step, changed_valid, graph_data=graph_data)

        assert torch.allclose(baseline, masked_same, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(baseline, valid_changed)


def test_masked_context_attention_ablation_reports_attention_pair_metrics():
    baseline = create_discrete_masked_model(context_dim=8, hidden_dim=16, context_attention_mode="concat_encoder")
    ablation = create_discrete_masked_model(context_dim=8, hidden_dim=16, context_attention_mode="cross_decoder")

    base_metrics = baseline.attention_complexity_metrics(context_tokens=64)
    ablation_metrics = ablation.attention_complexity_metrics(context_tokens=64)

    assert baseline.context_attention_mode == "concat_encoder"
    assert ablation.context_attention_mode == "cross_decoder"
    assert base_metrics["total_attention_pairs"] == base_metrics["baseline_concat_attention_pairs"]
    assert ablation_metrics["total_attention_pairs"] < base_metrics["total_attention_pairs"]
    assert 0.0 < ablation_metrics["relative_to_concat"] < 1.0


def test_masked_room_token_conversion_accepts_raw_and_normalized_maps():
    raw = torch.tensor([[[[0.0, 1.0], [17.0, 43.0]]]])
    normalized = raw / 43.0

    raw_ids = MaskedRoomTrainer._to_token_ids(raw, num_classes=44)
    normalized_ids = MaskedRoomTrainer._to_token_ids(normalized, num_classes=44)

    expected = torch.tensor([[[0, 1], [17, 43]]])
    assert torch.equal(raw_ids, expected)
    assert torch.equal(normalized_ids, expected)


def test_pipeline_generate_room_uses_discrete_masked_mode(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
        default_masked_room_sampling_temperature=0.9,
        default_masked_room_sampling_schedule="cosine",
        default_masked_room_sampling_stochastic=True,
        default_masked_room_corrector_steps=1,
        default_masked_room_corrector_mask_ratio=0.125,
    )
    pipeline.diffusion = None
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    called = {"sample": 0}

    def _sample(**kwargs):
        called["sample"] += 1
        called["kwargs"] = dict(kwargs)
        fixed_tokens = kwargs.get("fixed_tokens")
        fixed_mask = kwargs.get("fixed_mask")
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        if isinstance(fixed_tokens, torch.Tensor) and isinstance(fixed_mask, torch.Tensor):
            tokens[fixed_mask] = fixed_tokens[fixed_mask]
        logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert called["sample"] == 1
    assert called["kwargs"]["temperature"] == pytest.approx(0.9)
    assert called["kwargs"]["schedule_mode"] == "cosine"
    assert called["kwargs"]["stochastic"] is True
    assert called["kwargs"]["corrector_steps"] == 1
    assert called["kwargs"]["corrector_mask_ratio"] == pytest.approx(0.125)
    assert result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["START"]))) == 1
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["TRIFORCE"]))) == 0


def test_discrete_masked_model_sample_is_seeded_when_stochastic(monkeypatch):
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=8,
        num_steps=3,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=1,
    )
    context = torch.zeros(1, 1, 8)

    def _forward(tokens, step, context, *, graph_data=None, return_hidden=False):
        _ = (tokens, step, context, graph_data)
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-8.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 0.0
        logits[:, int(SEMANTIC_PALETTE["BLOCK"]), :, :] = 0.0
        hidden = torch.zeros(1, model.hidden_dim, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return (logits, hidden) if return_hidden else logits

    monkeypatch.setattr(model, "forward", _forward)

    sample_a, _logits_a, _hidden_a = model.sample(
        context=context,
        num_steps=3,
        stochastic=True,
        corrector_steps=0,
        seed=11,
    )
    sample_b, _logits_b, _hidden_b = model.sample(
        context=context,
        num_steps=3,
        stochastic=True,
        corrector_steps=0,
        seed=29,
    )
    sample_c, _logits_c, _hidden_c = model.sample(
        context=context,
        num_steps=3,
        stochastic=True,
        corrector_steps=0,
        seed=11,
    )

    assert not torch.equal(sample_a, sample_b)
    assert torch.equal(sample_a, sample_c)


def test_discrete_masked_model_sample_can_disable_stochastic_decode(monkeypatch):
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=8,
        num_steps=3,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=1,
    )
    context = torch.zeros(1, 1, 8)

    def _forward(tokens, step, context, *, graph_data=None, return_hidden=False):
        _ = (tokens, step, context, graph_data)
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-8.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["BLOCK"]), :, :] = 1.0
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 2.0
        hidden = torch.zeros(1, model.hidden_dim, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return (logits, hidden) if return_hidden else logits

    monkeypatch.setattr(model, "forward", _forward)

    sample_a, _logits_a, _hidden_a = model.sample(
        context=context,
        num_steps=3,
        stochastic=False,
        corrector_steps=0,
        seed=3,
    )
    sample_b, _logits_b, _hidden_b = model.sample(
        context=context,
        num_steps=3,
        stochastic=False,
        corrector_steps=0,
        seed=97,
    )

    assert torch.equal(sample_a, sample_b)
    assert int(sample_a[0, 4, 4]) == int(SEMANTIC_PALETTE["FLOOR"])


def test_pipeline_generate_room_masked_mode_falls_back_to_diffusion_teacher_on_noise(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
        default_masked_room_teacher_fallback_enabled=True,
    )
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    def _noisy_masked_sample(**kwargs):
        _ = kwargs
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        tokens[:, 4:12, 4:7] = int(SEMANTIC_PALETTE["BLOCK"])
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        logits[:, int(SEMANTIC_PALETTE["BLOCK"]), 4:12, 4:7] = 8.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    def _teacher_ddim_sample(*, context, shape, num_steps, graph_data=None):
        _ = (context, num_steps, graph_data)
        return torch.zeros(shape, dtype=torch.float32)

    def _teacher_decode(_latent):
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        return logits

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _noisy_masked_sample)
    monkeypatch.setattr(pipeline.diffusion, "ddim_sample", _teacher_ddim_sample)
    monkeypatch.setattr(pipeline.vqvae, "decode", _teacher_decode)
    sync_calls = {"count": 0}

    def _sync():
        sync_calls["count"] += 1

    monkeypatch.setattr(pipeline, "_synchronize_cuda_device", _sync)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert result.metrics["teacher_fallback_used"] == 1.0
    assert result.metrics["teacher_fallback_source_masked_room"] == 1.0
    assert sync_calls["count"] == 1
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["BLOCK"]))) == 0


def test_masked_room_fixed_tokens_only_emit_start_and_goal_in_true_role_rooms():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True, pos=(0, 0))
    graph.add_node(1, is_goal=True, pos=(0, 1))
    graph.add_edge(0, 1)

    start_tokens, start_mask = pipeline._build_masked_room_fixed_tokens(
        graph,
        0,
        start_goal=((8, 0), (8, 10)),
    )
    goal_tokens, goal_mask = pipeline._build_masked_room_fixed_tokens(
        graph,
        1,
        start_goal=((8, 0), (8, 10)),
    )

    start_role_flags = pipeline._room_role_flags(dict(graph.nodes[0]))
    start_semantics = pipeline._extract_room_topology_semantics(graph, 0)
    start_anchors = build_room_semantic_anchor_points(
        start=(8, 0),
        goal=(8, 10),
        required_doors=start_semantics["required_doors"],
        incoming_dirs=start_semantics["incoming_dirs"],
        outgoing_dirs=start_semantics["outgoing_dirs"],
        room_role_flags=start_role_flags,
    )
    sr, sc = start_anchors["start"]
    assert bool(start_mask[0, sr, sc])
    assert int(start_tokens[0, sr, sc]) == int(SEMANTIC_PALETTE["START"])

    goal_role_flags = pipeline._room_role_flags(dict(graph.nodes[1]))
    goal_semantics = pipeline._extract_room_topology_semantics(graph, 1)
    goal_anchors = build_room_semantic_anchor_points(
        start=(8, 0),
        goal=(8, 10),
        required_doors=goal_semantics["required_doors"],
        incoming_dirs=goal_semantics["incoming_dirs"],
        outgoing_dirs=goal_semantics["outgoing_dirs"],
        room_role_flags=goal_role_flags,
    )
    gr, gc = goal_anchors["goal"]
    assert bool(goal_mask[0, gr, gc])
    assert int(goal_tokens[0, gr, gc]) == int(SEMANTIC_PALETTE["TRIFORCE"])


def test_masked_room_fixed_tokens_place_shared_semantic_anchors_for_graph_roles():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    graph = nx.DiGraph()
    graph.add_node(0, label="BIG_KEY", type="BIG_KEY", has_key=True, pos=(0, 0))
    graph.add_node(1, label="ITEM", type="ITEM", has_item=True, pos=(0, 1))
    graph.add_node(2, label="SWITCH", type="SWITCH", has_puzzle=True, pos=(0, 2))
    graph.add_node(3, label="BOSS", type="BOSS", has_boss=True, pos=(0, 3))

    for room_id, anchor_name, tile_name in (
        (0, "key", "KEY_BOSS"),
        (1, "item", "KEY_ITEM"),
        (2, "puzzle", "PUZZLE"),
        (3, "boss", "BOSS"),
    ):
        fixed_tokens, fixed_mask = pipeline._build_masked_room_fixed_tokens(
            graph,
            room_id,
            start_goal=((8, 0), (8, 10)),
        )
        role_flags = pipeline._room_role_flags(dict(graph.nodes[room_id]))
        semantics = pipeline._extract_room_topology_semantics(graph, room_id)
        anchors = build_room_semantic_anchor_points(
            start=(8, 0),
            goal=(8, 10),
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
        )
        rr, cc = anchors[anchor_name]
        assert bool(fixed_mask[0, rr, cc])
        assert int(fixed_tokens[0, rr, cc]) == int(SEMANTIC_PALETTE[tile_name])


def test_topology_fixed_mask_keeps_semantic_role_anchors_during_training():
    tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
    topo = torch.zeros(1, 50, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)

    key_anchor = (5, 3)
    boss_anchor = (10, 7)
    topo[0, ROOM_TOPOLOGY_CHANNELS["role_key"], key_anchor[0], key_anchor[1]] = 1.0
    topo[0, ROOM_TOPOLOGY_CHANNELS["role_boss"], boss_anchor[0], boss_anchor[1]] = 1.0
    tokens[0, key_anchor[0], key_anchor[1]] = int(SEMANTIC_PALETTE["KEY_BOSS"])
    tokens[0, boss_anchor[0], boss_anchor[1]] = int(SEMANTIC_PALETTE["BOSS"])

    fixed_tokens, fixed_mask = DiscreteMaskedRoomModel.build_fixed_mask_from_topology_map(
        tokens,
        topo,
        num_classes=44,
    )

    assert bool(fixed_mask[0, key_anchor[0], key_anchor[1]])
    assert bool(fixed_mask[0, boss_anchor[0], boss_anchor[1]])
    assert int(fixed_tokens[0, key_anchor[0], key_anchor[1]]) == int(SEMANTIC_PALETTE["KEY_BOSS"])
    assert int(fixed_tokens[0, boss_anchor[0], boss_anchor[1]]) == int(SEMANTIC_PALETTE["BOSS"])


def test_overlay_room_graph_markers_maps_generated_graph_types_to_semantic_tiles():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    base_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)

    graph = nx.DiGraph()
    graph.add_node(0, label="BIG_KEY", type="BIG_KEY", has_key=True, pos=(0, 0))
    graph.add_node(1, label="STAIRS_DOWN", type="STAIRS_DOWN", pos=(0, 1))
    graph.add_node(2, label="SWITCH", type="SWITCH", pos=(0, 2))

    key_grid, key_count, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )
    stair_grid, stair_count, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=graph,
        room_id=1,
        start_goal=((8, 0), (8, 10)),
    )
    puzzle_grid, puzzle_count, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=graph,
        room_id=2,
        start_goal=((8, 0), (8, 10)),
    )

    assert key_count == 1
    assert int(np.sum(key_grid == int(SEMANTIC_PALETTE["KEY_BOSS"]))) == 1

    assert stair_count == 1
    assert int(np.sum(stair_grid == int(SEMANTIC_PALETTE["STAIR"]))) == 1

    assert puzzle_count == 1
    assert int(np.sum(puzzle_grid == int(SEMANTIC_PALETTE["PUZZLE"]))) == 1
    assert int(np.sum(puzzle_grid == int(SEMANTIC_PALETTE["START"]))) == 0


def test_room_role_flags_do_not_treat_switch_label_as_start():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )

    role_flags = pipeline._room_role_flags({"label": "S", "type": "SWITCH", "has_puzzle": True})

    assert role_flags["is_start"] is False
    assert role_flags["has_puzzle"] is True


def test_pipeline_generate_room_strips_hallucinated_semantics_and_replays_graph_markers(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
        default_masked_room_teacher_fallback_enabled=False,
    )
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, label="GOAL", type="GOAL", is_goal=True, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    def _sample(**kwargs):
        _ = kwargs
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        tokens[:, 5, 2] = int(SEMANTIC_PALETTE["ENEMY"])
        tokens[:, 6, 3] = int(SEMANTIC_PALETTE["PUZZLE"])
        tokens[:, 7, 4] = int(SEMANTIC_PALETTE["STAIR"])
        tokens[:, 8, 5] = int(SEMANTIC_PALETTE["ELEMENT"])
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        logits[:, int(SEMANTIC_PALETTE["ENEMY"]), 5, 2] = 7.0
        logits[:, int(SEMANTIC_PALETTE["PUZZLE"]), 6, 3] = 7.0
        logits[:, int(SEMANTIC_PALETTE["STAIR"]), 7, 4] = 7.0
        logits[:, int(SEMANTIC_PALETTE["ELEMENT"]), 8, 5] = 7.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["TRIFORCE"]))) == 1
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["ENEMY"]))) == 0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["PUZZLE"]))) == 0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["STAIR"]))) == 0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["ELEMENT"]))) == 0
    assert result.metrics["neural_semantic_tiles_stripped"] >= 3
    assert result.metrics["final_graph_markers_placed"] == 1


def test_discrete_masked_model_accepts_unbatched_edge_index_graph_context():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=8,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=1,
    )
    tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=44, dtype=torch.long)
    step = torch.zeros(1, dtype=torch.long)
    context = torch.randn(1, 2, 8)
    graph_data = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "node_positions": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        "current_node_distance": torch.randn(2, 4),
    }

    logits = model.forward(tokens, step, context, graph_data=graph_data)

    assert tuple(logits.shape) == (1, 44, ROOM_HEIGHT, ROOM_WIDTH)


def test_masked_room_trainer_passes_configurable_mask_schedule(monkeypatch):
    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        min_mask_ratio=0.25,
        max_mask_ratio=0.55,
        topology_alignment_weight=0.4,
        topology_marker_weight=2.5,
        topology_trace_weight=0.9,
        topology_focus_dilation=2,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )
    trainer = MaskedRoomTrainer(config)
    captured = {}

    def _fake_training_loss(*args, **kwargs):
        captured["min_mask_ratio"] = kwargs.get("min_mask_ratio")
        captured["max_mask_ratio"] = kwargs.get("max_mask_ratio")
        captured["topology_alignment_weight"] = kwargs.get("topology_alignment_weight")
        captured["topology_focus_map"] = kwargs.get("topology_focus_map")
        return torch.tensor(0.0, device=trainer.device), {
            "loss": 0.0,
            "base_loss": 0.0,
            "mask_ratio": 0.0,
            "masked_fraction": 0.0,
            "topology_focus_loss": 0.0,
            "topology_focus_fraction": 0.0,
        }

    monkeypatch.setattr(trainer.model, "training_loss", _fake_training_loss)

    real_maps = torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    metrics = trainer._step(real_maps, graph_list=None, train=False)

    assert metrics["loss"] == 0.0
    assert captured["min_mask_ratio"] == 0.25
    assert captured["max_mask_ratio"] == 0.55
    assert captured["topology_alignment_weight"] == pytest.approx(0.4)
    assert captured["topology_focus_map"] is None


def _masked_room_logic_graph(current_node_idx: int) -> dict:
    node_features = torch.zeros(2, 6, dtype=torch.float32)
    node_features[0, 0] = 1.0
    node_features[1, 3] = 1.0
    return {
        "node_features": node_features,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.zeros(2, dtype=torch.long),
        "num_nodes": 2,
        "node_to_idx": {"A": 0, "B": 1},
        "current_node_idx": int(current_node_idx),
        "start_node_id": 0,
        "target_idx": 1,
        "room_topology_map": torch.zeros(54, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
    }


def test_masked_room_trainer_logic_ablation_uses_raw_logits(monkeypatch):
    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        logic_net_enabled=True,
        alpha_logic=0.5,
        logic_net_trainable=False,
        num_logic_iterations=2,
        hidden_dim=32,
        context_dim=8,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )
    trainer = MaskedRoomTrainer(config)

    def _fake_encode(_graph_dict):
        return torch.zeros(2, config.context_dim, dtype=torch.float32)

    monkeypatch.setattr(trainer, "_encode_graph_conditioning", _fake_encode)

    captured = {}

    def _fake_training_loss(*args, **kwargs):
        _ = args
        assert kwargs.get("return_aux") is True
        logits = torch.full((2, config.num_classes, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32)
        captured["graph_scope_for_model"] = kwargs["graph_data"]["graph_scope"]
        return torch.tensor(1.0), {
            "loss": 1.0,
            "base_loss": 1.0,
            "mask_ratio": 0.5,
            "masked_fraction": 0.5,
            "topology_focus_loss": 0.0,
            "topology_focus_fraction": 0.0,
        }, {"logits": logits}

    monkeypatch.setattr(trainer.model, "training_loss", _fake_training_loss)

    class _FakeLogicNet(torch.nn.Module):
        def forward(self, tile_logits, graph_data=None):
            captured["logic_input_shape"] = tuple(tile_logits.shape)
            captured["graph_scope_for_logic"] = graph_data["graph_scope"]
            return tile_logits.mean() * 0.0 + torch.tensor(0.25), {
                "global_graph_reachability": torch.tensor(0.75),
                "global_graph_node_coverage": 1.0,
                "global_room_passability": torch.tensor(0.5),
            }

    trainer.logic_net = _FakeLogicNet()
    real_maps = torch.zeros(2, 1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    graph_list = [_masked_room_logic_graph(0), _masked_room_logic_graph(1)]

    metrics = trainer._step(real_maps, graph_list, train=False)

    assert captured["graph_scope_for_model"] == "dungeon"
    assert captured["graph_scope_for_logic"] == "dungeon"
    assert captured["logic_input_shape"] == (2, config.num_classes, ROOM_HEIGHT, ROOM_WIDTH)
    assert metrics["loss"] == pytest.approx(1.125)
    assert metrics["logic_loss"] == pytest.approx(0.25)
    assert metrics["logic_loss_contribution"] == pytest.approx(0.125)
    assert metrics["logic_global_graph_supervised"] == pytest.approx(1.0)
    assert metrics["logic_global_graph_loss_skipped"] == pytest.approx(0.0)
    assert metrics["logic_global_graph_node_coverage"] == pytest.approx(1.0)
    assert metrics["logic_global_graph_reachability"] == pytest.approx(0.75)


def test_discrete_masked_model_training_loss_reports_topology_focus_term():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=8,
        num_steps=3,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=1,
    )
    context = torch.zeros(1, 1, 8)
    target_tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    topology_focus_map = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    topology_focus_map[:, 4:8, 3:7] = 1.0

    loss, metrics = model.training_loss(
        target_tokens,
        context,
        min_mask_ratio=1.0,
        max_mask_ratio=1.0,
        topology_focus_map=topology_focus_map,
        topology_alignment_weight=0.5,
    )

    assert float(loss.item()) >= 0.0
    assert metrics["base_loss"] >= 0.0
    assert metrics["topology_focus_loss"] >= 0.0
    assert metrics["topology_focus_fraction"] > 0.0


def test_edge_aware_logit_bias_skips_missing_topology_channels():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        context_dim=8,
        num_steps=3,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=1,
    )
    logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    context = {"room_topology_map": torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH)}

    biased = model._apply_edge_aware_logit_bias(logits, context)

    assert torch.allclose(biased, logits)


def test_masked_room_trainer_can_pass_reference_room_maps_into_condition_encoder():
    trainer = MaskedRoomTrainer.__new__(MaskedRoomTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = type(
        "Cfg",
        (),
        {
            "graph_conditioning_mode": "node_sequence",
            "condition_use_reference_room_maps": True,
            "use_current_node_distance_features": True,
            "current_node_distance_max": 8,
        },
    )()
    trainer.condition_encoder = _DummyMaskedConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        "room_position": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "current_node_idx": 0,
        "neighbor_maps": {
            "N": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32),
            "S": None,
            "E": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.75, dtype=torch.float32),
            "W": None,
        },
    }

    _encoded = MaskedRoomTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_reference_room_maps is graph_dict["neighbor_maps"]


def test_masked_room_trainer_passes_explicit_style_id_into_condition_encoder():
    trainer = MaskedRoomTrainer.__new__(MaskedRoomTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = type(
        "Cfg",
        (),
        {
            "graph_conditioning_mode": "node_sequence",
            "condition_use_reference_room_maps": False,
            "use_current_node_distance_features": True,
            "current_node_distance_max": 8,
        },
    )()
    trainer.condition_encoder = _DummyMaskedConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        "room_position": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "current_node_idx": 0,
        "style_id": 3,
    }

    _encoded = MaskedRoomTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_style_id == 3


def test_masked_room_trainer_appends_stage_tokens_when_enabled():
    trainer = MaskedRoomTrainer.__new__(MaskedRoomTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = type(
        "Cfg",
        (),
        {
            "graph_conditioning_mode": "node_sequence",
            "condition_use_reference_room_maps": False,
            "use_current_node_distance_features": True,
            "current_node_distance_max": 8,
            "puzzle_stage_conditioning_enabled": True,
            "puzzle_stage_token_scale": 0.20,
        },
    )()
    trainer.condition_encoder = _DummyMaskedConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        "room_position": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "current_node_idx": 0,
        "puzzle_stage_condition": {
            "gate_family": "key",
            "sequence_required": True,
            "controlled_doors": ["E"],
            "stage_sequence": [
                {"stage_index": 0, "kind": "collect_key", "local_anchor": [5, 4]},
                {"stage_index": 1, "kind": "reach_exit", "local_anchor": [5, 9]},
            ],
        },
    }

    encoded = MaskedRoomTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert tuple(encoded.shape) == (5, 8)


def test_masked_room_resume_checkpoint_round_trip(tmp_path):
    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        checkpoint_dir=str(tmp_path),
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )
    trainer = MaskedRoomTrainer(config)
    trainer.epoch = 3
    trainer.global_step = 17

    tracked_param = next(trainer.model.parameters())
    original = tracked_param.detach().clone()

    resume_path = tmp_path / "masked_room_resume.pth"
    inference_path = tmp_path / "masked_room_inference.pth"

    trainer.save_checkpoint(str(resume_path), {"val_loss": 1.25}, include_optimizer=True)
    resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in resume_payload
    assert "scheduler_state_dict" in resume_payload
    assert resume_payload["metadata"]["topology_anchor_policy"]["version"]
    assert resume_payload["epoch"] == 3
    assert resume_payload["global_step"] == 17

    trainer.save_checkpoint(str(inference_path), {"val_loss": 1.25}, include_optimizer=False)
    inference_payload = torch.load(inference_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in inference_payload
    assert "scheduler_state_dict" not in inference_payload
    assert inference_payload["metadata"]["topology_anchor_policy"]["version"]

    with torch.no_grad():
        tracked_param.zero_()
    trainer.epoch = 0
    trainer.global_step = 0

    trainer.load_checkpoint(str(resume_path))

    assert trainer.epoch == 3
    assert trainer.global_step == 17
    assert torch.allclose(tracked_param, original)


def test_masked_room_dataloaders_use_real_validation_split(monkeypatch):
    sample = (torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32), {"room_topology_map": torch.zeros(54, ROOM_HEIGHT, ROOM_WIDTH)})
    fake_dataset = [sample for _ in range(10)]

    def _fake_create_dataloader(*args, **kwargs):
        _ = (args, kwargs)
        return _DatasetLoaderStub(fake_dataset)

    monkeypatch.setattr("src.train_masked_room.create_dataloader", _fake_create_dataloader)

    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        batch_size=2,
        validation_fraction=0.2,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )

    train_loader, val_loader, eval_split_name, train_size, eval_size = _create_masked_room_dataloaders(config)

    assert eval_split_name == "val"
    assert train_size == 8
    assert eval_size == 2
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2


def test_masked_room_logic_dataloaders_use_dungeon_batch_sampler(monkeypatch):
    class _FakeRoomDataset:
        def __init__(self):
            self.sample_metadata = [
                {"dungeon_id": "d0", "current_node_idx": 0},
                {"dungeon_id": "d0", "current_node_idx": 1},
                {"dungeon_id": "d1", "current_node_idx": 0},
                {"dungeon_id": "d1", "current_node_idx": 1},
            ]

        def __len__(self):
            return len(self.sample_metadata)

        def __getitem__(self, idx):
            graph = {
                "node_features": torch.zeros(2, 6, dtype=torch.float32),
                "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                "num_nodes": 2,
                "node_to_idx": {"A": 0, "B": 1},
                "current_node_idx": int(self.sample_metadata[idx]["current_node_idx"]),
            }
            return torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32), graph

    monkeypatch.setattr(
        "src.train_masked_room.create_dataloader",
        lambda *args, **kwargs: _DatasetLoaderStub(_FakeRoomDataset()),
    )
    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        batch_size=2,
        validation_fraction=0.0,
        logic_net_enabled=True,
        alpha_logic=0.5,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )

    train_loader, val_loader, eval_split_name, train_size, eval_size = _create_masked_room_dataloaders(config)

    assert isinstance(train_loader.batch_sampler, DungeonBatchSampler)
    assert isinstance(val_loader.batch_sampler, DungeonBatchSampler)
    assert eval_split_name == "train"
    assert train_size == 4
    assert eval_size == 4


def test_masked_room_auto_resume_skips_incompatible_latest_checkpoint(
    monkeypatch,
    tmp_path,
    caplog,
):
    latest_resume = tmp_path / "latest_resume.pth"
    latest_resume.write_bytes(b"stub")

    class _EmptyLoader:
        def __init__(self):
            self.dataset = [0]

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    monkeypatch.setattr("src.train_masked_room.create_dataloader", lambda *args, **kwargs: _EmptyLoader())

    def _boom(self, path: str):
        raise RuntimeError("size mismatch for masked-room checkpoint")

    monkeypatch.setattr(MaskedRoomTrainer, "load_checkpoint", _boom)

    config = MaskedRoomTrainingConfig(
        device="cpu",
        data_dir="unused",
        checkpoint_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
        auto_resume=True,
    )

    caplog.set_level(logging.WARNING)
    trainer = train_masked_room(config)

    assert trainer.epoch == 0
    assert "Skipping auto-resume masked-room checkpoint" in caplog.text
