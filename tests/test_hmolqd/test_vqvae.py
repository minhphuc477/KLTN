"""
Tests for H-MOLQD Block II: Semantic VQ-VAE
============================================

Tests for vector quantization, encoder/decoder, and training.
"""

import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch")


class TestVectorQuantizer:
    """Tests for Vector Quantizer module."""
    
    def test_quantizer_forward(self):
        """Test quantizer forward pass."""
        from src.core.vqvae import VectorQuantizer
        
        quantizer = VectorQuantizer(
            num_embeddings=64,
            embedding_dim=32,
        )
        
        # Random input
        x = torch.randn(2, 32, 8, 8)  # B x C x H x W
        
        quantized, loss, indices = quantizer(x)
        
        assert quantized.shape == x.shape
        assert loss.ndim == 0  # Scalar
        assert indices.shape == (2, 8, 8)
    
    def test_quantizer_codebook_usage(self):
        """Test that codebook is being used."""
        from src.core.vqvae import VectorQuantizer
        
        quantizer = VectorQuantizer(
            num_embeddings=16,
            embedding_dim=8,
        )
        
        # Forward pass with diverse input
        x = torch.randn(10, 8, 4, 4)
        _, _, indices = quantizer(x)
        
        # Check that multiple codes are used
        unique_codes = torch.unique(indices)
        assert len(unique_codes) >= 2

    def test_dead_code_reset_respects_warmup(self):
        """Dead-code resets should not fire before the configured warmup."""
        from src.core.vqvae import VectorQuantizer

        quantizer = VectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            dead_code_threshold=0.05,
            dead_code_warmup_steps=10,
        )

        before = quantizer.embedding.weight.detach().clone()
        quantizer._reset_counter = 5
        z_flat = torch.randn(6, 4)
        indices = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

        quantizer._reset_dead_codes(z_flat, indices)

        assert torch.allclose(quantizer.embedding.weight, before)

    def test_dead_code_reset_protects_batch_active_codes(self):
        """Codes active in the current batch should not be reset."""
        from src.core.vqvae import VectorQuantizer

        quantizer = VectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            dead_code_threshold=0.05,
            dead_code_warmup_steps=0,
            protect_active_codes_during_reset=True,
        )
        quantizer.ema_cluster_size.zero_()
        before = quantizer.embedding.weight.detach().clone()
        quantizer._reset_counter = quantizer._reset_interval
        z_flat = torch.randn(6, 4)
        indices = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

        quantizer._reset_dead_codes(z_flat, indices)

        assert torch.allclose(quantizer.embedding.weight[:4], before[:4])
        assert not torch.allclose(quantizer.embedding.weight[4:], before[4:])

    def test_dead_code_reset_is_capped_per_event(self):
        """Large dead-code sets should be reset gradually, not all at once."""
        from src.core.vqvae import VectorQuantizer

        quantizer = VectorQuantizer(
            num_embeddings=32,
            embedding_dim=4,
            dead_code_threshold=0.05,
            dead_code_warmup_steps=0,
            protect_active_codes_during_reset=False,
            max_dead_code_resets_per_event=3,
        )
        quantizer.ema_cluster_size.zero_()
        before = quantizer.embedding.weight.detach().clone()
        quantizer._reset_counter = quantizer._reset_interval
        z_flat = torch.randn(12, 4)
        indices = torch.zeros(12, dtype=torch.long)

        quantizer._reset_dead_codes(z_flat, indices)

        changed = (~torch.isclose(quantizer.embedding.weight, before)).any(dim=1).sum().item()
        assert changed == 3

    def test_ema_quantizer_clamps_smoothed_cluster_denominator(self):
        """Dead-code EMA normalization should not create NaN/Inf embeddings."""
        from src.core.vqvae import VectorQuantizer

        quantizer = VectorQuantizer(
            num_embeddings=4,
            embedding_dim=3,
            use_ema=True,
            epsilon=1e-5,
            dead_code_reset_interval=1000,
        )
        quantizer.train()
        quantizer.ema_cluster_size.zero_()
        quantizer.ema_embedding_sum.fill_(1.0)

        z_e = torch.zeros(1, 3, 1, 1)
        quantizer(z_e)

        assert torch.isfinite(quantizer.embedding.weight).all()

    def test_quantizer_deepcopy_recreates_update_lock(self):
        """The thread lock must not make the quantizer uncopyable."""
        import copy
        from src.core.vqvae import VectorQuantizer

        quantizer = VectorQuantizer(num_embeddings=4, embedding_dim=3)
        copied = copy.deepcopy(quantizer)

        assert copied._codebook_update_lock is not quantizer._codebook_update_lock


class TestFSQuantizer:
    def test_fsq_quantizer_forward_indices_and_gradients(self):
        from src.core.vqvae import FSQuantizer

        quantizer = FSQuantizer(embedding_dim=8, levels=[4, 4, 4, 4])
        x = torch.randn(2, 8, 3, 4, requires_grad=True)

        quantized, indices, losses = quantizer(x, return_info=True)
        loss = quantized.mean() + losses["vq_loss"]
        loss.backward()

        assert tuple(quantized.shape) == tuple(x.shape)
        assert tuple(indices.shape) == (2, 3, 4)
        assert losses["vq_loss"].item() == pytest.approx(0.0)
        assert losses["commitment_loss"].item() == pytest.approx(0.0)
        assert quantizer.num_embeddings == 256
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_create_vqvae_fsq_architecture_round_trip(self):
        from src.core.vqvae import FSQuantizer, create_vqvae

        model = create_vqvae(
            num_classes=5,
            codebook_size=256,
            latent_dim=8,
            hidden_dim=16,
            architecture="fsq",
        )
        x = torch.zeros(1, 5, 16, 11)
        x[:, 1] = 1.0
        recon, _vq_loss, losses = model(x)

        assert isinstance(model.quantizer, FSQuantizer)
        assert tuple(recon.shape) == (1, 5, 16, 11)
        assert losses["vq_loss"].item() == pytest.approx(0.0)


class TestEncoder:
    """Tests for VQ-VAE Encoder."""
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        from src.core.vqvae import Encoder
        
        encoder = Encoder(
            in_channels=44,  # Tile classes
            hidden_dims=[32, 64, 128],
            latent_dim=64,
        )
        
        # One-hot encoded input
        x = torch.randn(2, 44, 16, 11)  # B x C x H x W
        
        z = encoder(x)
        
        assert z.ndim == 4
        assert z.shape[1] == 64  # latent_dim
    
    def test_encoder_spatial_reduction(self):
        """Test that encoder reduces spatial dimensions."""
        from src.core.vqvae import Encoder
        
        encoder = Encoder(
            in_channels=44,
            hidden_dims=[32, 64],
            latent_dim=32,
        )
        
        x = torch.randn(1, 44, 16, 12)  # 16x12 input
        z = encoder(x)
        
        # Each layer halves spatial dims
        assert z.shape[2] < 16
        assert z.shape[3] < 12


class TestDecoder:
    """Tests for VQ-VAE Decoder."""
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        from src.core.vqvae import Decoder
        
        decoder = Decoder(
            out_channels=44,
            hidden_dims=[128, 64, 32],
            latent_dim=64,
        )
        
        z = torch.randn(2, 64, 4, 3)
        
        x_recon = decoder(z)
        
        assert x_recon.ndim == 4
        assert x_recon.shape[1] == 44


class TestSemanticVQVAE:
    """Tests for complete Semantic VQ-VAE."""

    def test_canonical_latent_shape_matches_default_encoder_output(self):
        """Shared latent-shape helper should match the default VQ-VAE encoder."""
        from src.core.vqvae import SemanticVQVAE, canonical_latent_shape

        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=32,
            num_embeddings=64,
        )

        x = torch.randn(1, 44, 16, 11)
        z_q, indices = model.encode(x)

        assert tuple(z_q.shape[-2:]) == canonical_latent_shape((16, 11))
        assert tuple(indices.shape[-2:]) == canonical_latent_shape((16, 11))
    
    def test_vqvae_forward(self):
        """Test VQ-VAE forward pass."""
        from src.core.vqvae import SemanticVQVAE
        
        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=32,
            num_embeddings=64,
            hidden_dims=[16, 32],
        )
        
        x = torch.randn(2, 44, 16, 11)
        
        recon, vq_loss, _indices = model(x)
        
        assert recon.shape == x.shape
        assert vq_loss.ndim == 0
    
    def test_vqvae_encode_decode(self):
        """Test encode and decode separately."""
        from src.core.vqvae import SemanticVQVAE
        
        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=32,
            num_embeddings=64,
            hidden_dims=[16, 32],
        )
        
        x = torch.randn(1, 44, 16, 11)
        
        z = model.encode(x)
        z_quantized, _, _ = model.quantize(z)
        recon = model.decode(z_quantized)
        
        assert recon.shape == x.shape

    def test_vqvae_no_codebook_forward(self):
        """Test plain autoencoder mode with the codebook disabled."""
        from src.core.vqvae import SemanticVQVAE

        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=32,
            num_embeddings=64,
            hidden_dims=[16, 32],
            use_codebook=False,
        )

        x = torch.randn(2, 44, 16, 11)

        recon, vq_loss, losses = model(x)

        assert recon.shape == x.shape
        assert vq_loss.ndim == 0
        assert torch.isclose(vq_loss, torch.zeros_like(vq_loss))
        assert torch.isclose(losses["perplexity"], torch.zeros_like(losses["perplexity"]))

        z = model.encode(x)
        z_quantized, quant_loss, indices = model.quantize(z)

        assert z_quantized.shape == z[0].shape
        assert quant_loss.ndim == 0
        assert torch.isclose(quant_loss, torch.zeros_like(quant_loss))
        assert indices.shape[-2:] == z[1].shape[-2:]
    
    def test_vqvae_loss_computation(self):
        """Test loss computation."""
        from src.core.vqvae import SemanticVQVAE
        
        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=32,
            num_embeddings=64,
            hidden_dims=[16, 32],
        )
        
        x = torch.randn(2, 44, 16, 11)
        
        loss_dict = model.compute_loss(x)
        
        assert 'total_loss' in loss_dict
        assert 'recon_loss' in loss_dict
        assert 'vq_loss' in loss_dict
        
        assert loss_dict['total_loss'] >= 0


class TestVQVAETrainer:
    """Tests for VQ-VAE Trainer."""
    
    def test_trainer_step(self):
        """Test single training step."""
        from src.core.vqvae import SemanticVQVAE, VQVAETrainer
        
        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=32,
            num_embeddings=64,
            hidden_dims=[16, 32],
        )
        
        trainer = VQVAETrainer(model, learning_rate=1e-3)
        
        batch = torch.randn(4, 44, 16, 11)
        
        loss = trainer.train_step(batch)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestVQVAETrainingHelpers:
    """Focused tests for Block II training/evaluation helpers."""

    def test_validation_split_is_deterministic_and_nonempty(self):
        """Validation split helper should create a stable held-out slice."""
        from src.train_vqvae import split_dataset_for_vqvae_validation

        dataset = list(range(20))
        train_a, val_a = split_dataset_for_vqvae_validation(dataset, validation_fraction=0.2, seed=123)
        train_b, val_b = split_dataset_for_vqvae_validation(dataset, validation_fraction=0.2, seed=123)

        assert len(train_a) == 16
        assert len(val_a) == 4
        assert train_a.indices == train_b.indices
        assert val_a.indices == val_b.indices

    def test_codebook_health_reports_utilization_metrics(self):
        """Codebook-health helper should expose activity/utilization fields."""
        from src.core.vqvae import SemanticVQVAE
        from src.train_vqvae import compute_vqvae_codebook_health

        model = SemanticVQVAE(
            num_tile_classes=44,
            latent_dim=16,
            num_embeddings=8,
            hidden_dims=[8, 16],
        )
        model.quantizer.codebook_usage.zero_()
        model.quantizer.codebook_usage[:3] = torch.tensor([1.0, 0.5, 0.25])
        if hasattr(model.quantizer, "ema_cluster_size"):
            model.quantizer.ema_cluster_size.zero_()
            model.quantizer.ema_cluster_size[:2] = torch.tensor([0.2, 0.1])

        metrics = compute_vqvae_codebook_health(model)

        assert metrics["codebook_size"] == 8.0
        assert metrics["codebook_active_codes"] == 3.0
        assert metrics["codebook_active_codes_gt_1e4"] == 3.0
        assert metrics["codebook_utilization"] == pytest.approx(3.0 / 8.0)
        assert "codebook_usage_entropy" in metrics
        assert "ema_live_codes" in metrics

    def test_vqvae2_forward_and_hierarchical_codebook_health(self):
        """Hierarchical VQ-VAE-2 should keep the downstream VQ-VAE interface."""
        from src.core.vqvae import create_vqvae
        from src.train_vqvae import compute_vqvae_codebook_health

        model = create_vqvae(
            architecture="vqvae2",
            num_classes=44,
            latent_dim=8,
            hidden_dim=8,
            codebook_size=8,
            top_codebook_size=8,
            top_latent_dim=8,
        )
        batch = torch.zeros(2, 44, 16, 11)
        batch[:, 1, :, :] = 1.0

        recon, vq_loss, losses = model(batch)
        z_q, indices = model.encode(batch)
        decoded = model.decode_indices(indices)

        assert recon.shape == batch.shape
        assert decoded.shape == batch.shape
        assert z_q.shape[1] == 8
        assert indices.shape[-2:] == z_q.shape[-2:]
        assert vq_loss.ndim == 0
        assert "top_vq_loss" in losses
        assert "bottom_vq_loss" in losses

        model.top_quantizer.codebook_usage.zero_()
        model.top_quantizer.codebook_usage[:2] = torch.tensor([1.0, 0.5])
        model.bottom_quantizer.codebook_usage.zero_()
        model.bottom_quantizer.codebook_usage[:4] = torch.tensor([1.0, 0.5, 0.25, 0.125])
        metrics = compute_vqvae_codebook_health(model)

        assert metrics["codebook_levels"] == 2.0
        assert metrics["top_codebook_utilization"] == pytest.approx(2.0 / 8.0)
        assert metrics["bottom_codebook_utilization"] == pytest.approx(4.0 / 8.0)
        assert metrics["codebook_utilization"] == pytest.approx(6.0 / 16.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
