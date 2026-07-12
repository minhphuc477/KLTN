from __future__ import annotations

import pytest
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.train_diffusion import DiffusionTrainingConfig
from src.train_lcm import FastSamplerTrainingConfig
from src.train_masked_room import MaskedRoomTrainingConfig
from src.train_vqvae import evaluate_vqvae_loader, split_dataset_for_vqvae_validation, train_vqvae
from src.zelda_data.splits import validate_disjoint_dungeon_splits
from src.zelda_data.zelda_loader import validate_floor_conditioning_signal


def test_split_validator_normalizes_and_preserves_disjoint_ids():
    train_ids, test_ids = validate_disjoint_dungeon_splits([3, 1, 3, 2], [9])

    assert train_ids == (1, 2, 3)
    assert test_ids == (9,)


def test_split_validator_rejects_holdout_overlap():
    with pytest.raises(ValueError, match="leak holdout data"):
        validate_disjoint_dungeon_splits([1, 2, 9], [9])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DiffusionTrainingConfig(train_dungeon_ids=[1, 9], test_dungeon_ids=[9]),
        lambda: MaskedRoomTrainingConfig(train_dungeon_ids=[1, 9], test_dungeon_ids=[9]),
        lambda: FastSamplerTrainingConfig(
            base_diffusion_checkpoint="unused-for-config-validation.pt",
            train_dungeon_ids=[1, 9],
            test_dungeon_ids=[9],
        ),
    ],
)
def test_training_configs_reject_holdout_overlap(factory):
    with pytest.raises(ValueError, match="leak holdout data"):
        factory()


def test_vqvae_entrypoint_rejects_holdout_overlap_before_loading_data():
    args = SimpleNamespace(
        epochs=1,
        quick=False,
        seed=None,
        verbose=False,
        device="cpu",
        train_dungeon_ids=[1, 9],
        test_dungeon_ids=[9],
    )

    with pytest.raises(ValueError, match="leak holdout data"):
        train_vqvae(args)


def test_internal_validation_split_groups_rooms_by_dungeon():
    class RoomDataset(Dataset):
        def __init__(self):
            self.sample_metadata = [
                {"dungeon_num": dungeon_num, "variant": variant}
                for dungeon_num in (1, 2, 3, 4)
                for variant in (1, 2)
                for _room in range(3)
            ]

        def __len__(self):
            return len(self.sample_metadata)

        def __getitem__(self, index):
            return index

    dataset = RoomDataset()
    train, validation = split_dataset_for_vqvae_validation(
        dataset,
        validation_fraction=0.25,
        seed=123,
    )

    assert validation is not None
    train_dungeons = {dataset.sample_metadata[index]["dungeon_num"] for index in train.indices}
    validation_dungeons = {
        dataset.sample_metadata[index]["dungeon_num"] for index in validation.indices
    }
    assert train_dungeons.isdisjoint(validation_dungeons)
    assert len(validation_dungeons) == 1


def test_vqvae_evaluation_restores_training_mode():
    model = torch.nn.Identity()
    model.train()
    trainer = SimpleNamespace(
        eval_step=lambda _batch: {
            "loss": 1.0,
            "recon_loss": 0.5,
            "illegal_adjacency_penalty": 0.0,
            "accuracy": 1.0,
            "perplexity": 2.0,
        }
    )
    loader = DataLoader(torch.zeros(2, 1, 2, 2), batch_size=1)

    metrics = evaluate_vqvae_loader(
        model,
        trainer,
        loader,
        num_classes=44,
        device=torch.device("cpu"),
    )

    assert model.training is True
    assert metrics["batches"] == 2.0


def test_floor_conditioning_rejects_single_floor_training_data():
    dataset = SimpleNamespace(
        graphs=[{
            "node_features": torch.zeros(3, 15),
            "floor_values": [0.0, 0.0, 0.0],
            "floor_labels_present": [True, True, True],
        }],
    )

    with pytest.raises(ValueError, match="fewer than two distinct floor labels"):
        validate_floor_conditioning_signal(dataset, node_feature_dim=15)


def test_floor_conditioning_accepts_observed_multi_floor_signal():
    node_features = torch.zeros(3, 15)
    node_features[2, 14] = 0.2
    dataset = SimpleNamespace(graphs=[{
        "node_features": node_features,
        "floor_values": [0.0, 0.0, 1.0],
        "floor_labels_present": [True, True, True],
    }])

    validate_floor_conditioning_signal(dataset, node_feature_dim=15)
    validate_floor_conditioning_signal(dataset, node_feature_dim=14)


def test_floor_conditioning_rejects_only_cross_dungeon_floor_variation():
    floor_zero = torch.zeros(2, 15)
    floor_one = torch.zeros(2, 15)
    floor_one[:, 14] = 0.2
    dataset = SimpleNamespace(graphs=[
        {
            "node_features": floor_zero,
            "floor_values": [0.0, 0.0],
            "floor_labels_present": [True, True],
        },
        {
            "node_features": floor_one,
            "floor_values": [1.0, 1.0],
            "floor_labels_present": [True, True],
        },
    ])

    with pytest.raises(ValueError, match="within any one dungeon"):
        validate_floor_conditioning_signal(dataset, node_feature_dim=15)
