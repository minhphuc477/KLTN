"""
Tests for the standalone Semantic Gaussian VAE baseline.
"""

import pytest

torch = pytest.importorskip("torch")


class TestSemanticGaussianVAE:
    def test_forward_and_shapes(self):
        from src.core.gaussian_vae import SemanticGaussianVAE
        from src.core.vqvae import canonical_latent_shape

        model = SemanticGaussianVAE(
            num_tile_classes=44,
            latent_dim=32,
            hidden_dim=16,
        )

        x = torch.randn(2, 44, 16, 11)
        recon, kl_loss, losses = model(x)

        assert recon.shape == x.shape
        assert kl_loss.ndim == 0
        assert losses["total_loss"].ndim == 0
        assert losses["kl_loss_weighted"].ndim == 0

        mu, logvar = model.encode(x)
        assert mu.shape == logvar.shape
        assert mu.shape[1] == 32
        assert tuple(mu.shape[-2:]) == canonical_latent_shape((16, 11))

    def test_sample_prior_matches_room_shape(self):
        from src.core.gaussian_vae import SemanticGaussianVAE

        model = SemanticGaussianVAE(
            num_tile_classes=44,
            latent_dim=16,
            hidden_dim=16,
        )

        sample = model.sample_prior(batch_size=3)

        assert sample.shape == (3, 44, 16, 11)


class TestGaussianVAETrainer:
    def test_trainer_step(self):
        from src.core.gaussian_vae import GaussianVAETrainer, SemanticGaussianVAE

        model = SemanticGaussianVAE(
            num_tile_classes=44,
            latent_dim=16,
            hidden_dim=16,
        )

        trainer = GaussianVAETrainer(model, learning_rate=1e-3)
        batch = torch.randn(4, 44, 16, 11)

        loss = trainer.train_step(batch)

        assert isinstance(loss, float)
        assert loss >= 0


class TestGaussianVAETrainingHelpers:
    def test_validation_split_is_deterministic_and_nonempty(self):
        from src.train_gaussian_vae import split_dataset_for_gaussian_vae_validation

        dataset = list(range(20))
        train_a, val_a = split_dataset_for_gaussian_vae_validation(
            dataset,
            validation_fraction=0.2,
            seed=123,
        )
        train_b, val_b = split_dataset_for_gaussian_vae_validation(
            dataset,
            validation_fraction=0.2,
            seed=123,
        )

        assert len(train_a) == 16
        assert len(val_a) == 4
        assert list(train_a.indices) == list(train_b.indices)
        assert list(val_a.indices) == list(val_b.indices)