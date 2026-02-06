"""
Tests for H-MOLQD Block II: Semantic VQ-VAE
============================================

Tests for vector quantization, encoder/decoder, and training.
"""

import pytest
import numpy as np

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
        
        recon, vq_loss, indices = model(x)
        
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
