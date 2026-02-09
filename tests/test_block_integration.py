"""
H-MOLQD Block Integration Test
================================

Validates that all 7 blocks can be instantiated and connected
end-to-end without import errors, signature mismatches, or crashes.

Run:
    python -m pytest tests/test_block_integration.py -v
    python tests/test_block_integration.py          # standalone
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


def test_block_ii_vqvae():
    """Block II: SemanticVQVAE instantiation, encode/decode round-trip."""
    from src.core.vqvae import SemanticVQVAE, create_vqvae

    model = create_vqvae(num_classes=44, codebook_size=64, latent_dim=32)
    x = torch.randn(2, 44, 11, 16)  # [B, C=44, H=11, W=16]

    # encode returns exactly 2 values
    z_q, indices = model.encode(x)
    assert z_q.shape[0] == 2
    assert z_q.shape[1] == 32  # latent_dim
    assert indices.shape[0] == 2

    # decode round-trip
    recon = model.decode(z_q, target_size=(11, 16))
    assert recon.shape == (2, 44, 11, 16)

    # full forward returns (recon, indices, losses_dict)
    recon, indices, losses = model(x)
    assert 'total_loss' in losses
    print("  ✓ Block II (VQ-VAE): encode/decode/forward OK")


def test_block_iii_condition_encoder():
    """Block III: DualStreamConditionEncoder with edge features."""
    from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder

    encoder = create_condition_encoder(latent_dim=32, output_dim=128)

    # Test encode_global_only (most common path)
    node_features = torch.randn(5, 5)
    edge_index = torch.tensor([[0,1,2,3,1], [1,2,3,4,3]], dtype=torch.long)
    edge_features = torch.randn(5, 8)  # Phase 3A: edge features

    c_global = encoder.encode_global_only(
        node_features, edge_index,
        edge_features=edge_features,
    )
    assert c_global.shape[0] == 5  # N nodes
    assert c_global.shape[1] == 128  # output_dim

    # Test without edge features (backward compatible)
    c_global_no_edge = encoder.encode_global_only(node_features, edge_index)
    assert c_global_no_edge.shape == c_global.shape
    print("  ✓ Block III (ConditionEncoder): global encoding with/without edge features OK")


def test_block_iv_latent_diffusion():
    """Block IV: LatentDiffusionModel training loss and sampling."""
    from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=32, model_channels=32, context_dim=64,
        num_timesteps=10,  # tiny for test speed
        cfg_scale=2.0,
        prediction_type='epsilon',
        min_snr_gamma=5.0,
    )

    z_0 = torch.randn(2, 32, 3, 4)
    context = torch.randn(2, 64)

    # Training loss
    loss = model.training_loss(z_0, context)
    assert loss.shape == ()
    assert not torch.isnan(loss)

    # DDPM sample
    with torch.no_grad():
        z_gen = model.sample(context, shape=(2, 32, 3, 4))
    assert z_gen.shape == (2, 32, 3, 4)

    # DDIM sample (checks CRITICAL-3 fix: alpha_t defined)
    with torch.no_grad():
        z_ddim = model.ddim_sample(context, shape=(2, 32, 3, 4), num_steps=5)
    assert z_ddim.shape == (2, 32, 3, 4)
    print("  ✓ Block IV (LatentDiffusion): loss, DDPM, DDIM sampling OK")


def test_block_v_logic_net():
    """Block V: LogicNet forward and temperature annealing."""
    from src.core.logic_net import LogicNet

    logic = LogicNet(latent_dim=32, num_classes=44, num_iterations=5)

    z = torch.randn(2, 32, 3, 4)  # latent codes
    loss, info = logic(z)
    assert loss.shape == ()
    assert 'walkability' in info

    # Temperature annealing (Phase 1D)
    logic.update_temperature(0.0)
    assert abs(logic.current_temperature.item() - 1.0) < 0.01
    logic.update_temperature(1.0)
    assert logic.current_temperature.item() < 0.1
    print("  ✓ Block V (LogicNet): forward + temperature annealing OK")


def test_block_vi_map_elites():
    """Block VI: MAP-Elites archive and feature extractors."""
    from src.evaluation.map_elites import (
        EliteArchive, CVTEliteArchive,
        CombinedFeatureExtractor, CBSFeatureExtractor, FullFeatureExtractor,
        create_map_elites,
    )

    # Standard archive
    archive = EliteArchive(feature_dims=2, cells_per_dim=5)
    assert archive.feature_dims == 2

    # CVT archive
    cvt = CVTEliteArchive(feature_dims=2, num_cells=10)
    assert cvt.num_cells == 10

    # Feature extractors
    combined = CombinedFeatureExtractor()
    assert callable(combined.extract)

    cbs = CBSFeatureExtractor()
    assert callable(cbs.extract)

    full = FullFeatureExtractor()
    assert callable(full.extract)

    # create_map_elites convenience
    me = create_map_elites(feature_type='combined', archive_type='grid')
    assert me is not None
    print("  ✓ Block VI (MAP-Elites): archives + extractors OK")


def test_block_vii_symbolic_refiner():
    """Block VII: SymbolicRefiner with LearnedTileStatistics."""
    from src.core.symbolic_refiner import (
        SymbolicRefiner, create_symbolic_refiner,
        LearnedTileStatistics, FailurePoint,
    )

    # Test FailurePoint with metadata (CRITICAL-5 fix)
    fp = FailurePoint(
        position=(5, 5),
        failure_type='blocked',
        required_item=None,
        metadata={'room_id': 3},
    )
    assert fp.metadata == {'room_id': 3}

    # LearnedTileStatistics (Phase 3B)
    stats = LearnedTileStatistics()
    fake_room = np.random.randint(0, 10, size=(11, 16))
    stats.observe(fake_room)
    assert stats._total_tiles > 0

    adj = stats.get_adjacency_rules(threshold=0.01)
    assert isinstance(adj, dict)

    weights = stats.get_tile_weights()
    assert len(weights) > 0

    # SymbolicRefiner with learned stats
    refiner = create_symbolic_refiner(learned_stats=stats)
    assert refiner.learned_stats is not None

    # Quick repair test
    grid = np.ones((11, 16), dtype=int)  # all floor
    grid[5, :] = 2  # wall barrier
    repaired, success = refiner.repair_room(grid, start=(2, 8), goal=(8, 8))
    assert repaired.shape == (11, 16)
    print("  ✓ Block VII (SymbolicRefiner): LearnedTileStatistics + repair OK")


def test_pipeline_vqvae_to_diffusion():
    """End-to-end: VQ-VAE encode → Diffusion loss → sample → VQ-VAE decode."""
    from src.core.vqvae import create_vqvae
    from src.core.latent_diffusion import create_latent_diffusion
    from src.core.condition_encoder import create_condition_encoder

    vqvae = create_vqvae(num_classes=44, codebook_size=32, latent_dim=16)
    diffusion = create_latent_diffusion(
        latent_dim=16, model_channels=16, context_dim=32,
        num_timesteps=10,
    )
    cond_encoder = create_condition_encoder(latent_dim=16, output_dim=32)

    # Simulate training step
    x = torch.randn(2, 44, 11, 16)
    vqvae.eval()
    with torch.no_grad():
        z_q, indices = vqvae.encode(x)  # CRITICAL-2: 2 values

    # Build conditioning
    node_features = torch.randn(4, 5)
    edge_index = torch.tensor([[0,1,2,1],[1,2,3,0]], dtype=torch.long)
    c_global = cond_encoder.encode_global_only(node_features, edge_index)
    conditioning = c_global.mean(dim=0, keepdim=True).expand(2, -1)

    # Diffusion training loss
    loss = diffusion.training_loss(z_q, conditioning)
    assert not torch.isnan(loss)

    # Sample and decode
    with torch.no_grad():
        z_gen = diffusion.ddim_sample(conditioning, shape=z_q.shape, num_steps=5)
        recon = vqvae.decode(z_gen, target_size=(11, 16))
    assert recon.shape == (2, 44, 11, 16)
    print("  ✓ Pipeline: VQ-VAE → Diffusion → VQ-VAE decode OK")


def test_trainer_instantiation():
    """DiffusionTrainer: can be constructed without crashes."""
    from src.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig

    config = DiffusionTrainingConfig(
        quick=True,
        epochs=1,
        num_timesteps=10,
        latent_dim=16,
        model_channels=16,
        context_dim=32,
    )
    # This exercises CRITICAL-1, CRITICAL-6 fixes
    trainer = DiffusionTrainer(config)
    assert trainer.vqvae is not None
    assert trainer.diffusion is not None
    assert trainer.condition_encoder is not None
    assert trainer.logic_net is not None
    assert trainer.ema_diffusion is not None  # Phase 4A

    # Check Block V LogicNet (not legacy)
    assert hasattr(trainer.logic_net, 'update_temperature'), \
        "Should be Block V LogicNet with temperature annealing, not legacy"
    print("  ✓ DiffusionTrainer: instantiation OK (all blocks connected)")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("H-MOLQD Block Integration Tests")
    print("=" * 60 + "\n")

    tests = [
        test_block_ii_vqvae,
        test_block_iii_condition_encoder,
        test_block_iv_latent_diffusion,
        test_block_v_logic_net,
        test_block_vi_map_elites,
        test_block_vii_symbolic_refiner,
        test_pipeline_vqvae_to_diffusion,
        test_trainer_instantiation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}\n")
