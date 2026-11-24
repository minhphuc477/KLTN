from src.zelda_data.zelda_loader import ZeldaRoomDataset, create_dataloader


def test_room_dataset_supports_dungeon_level_holdout_split():
    train = ZeldaRoomDataset(
        "Data/The Legend of Zelda",
        normalize=True,
        load_graphs=False,
        dungeon_ids=list(range(1, 9)),
        variants=[1, 2],
    )
    test = ZeldaRoomDataset(
        "Data/The Legend of Zelda",
        normalize=True,
        load_graphs=False,
        dungeon_ids=[9],
        variants=[1, 2],
    )

    # Wall/void-only framed slots are layout placeholders, not rooms.
    assert len(train) == 349
    assert len(test) == 101

    train_ids = {item["dungeon_id"] for item in train.sample_metadata}
    test_ids = {item["dungeon_id"] for item in test.sample_metadata}
    assert train_ids.isdisjoint(test_ids)
    assert test_ids == {"tloz9_1", "tloz9_2"}


def test_create_dataloader_forwards_dungeon_split_to_dataset():
    loader = create_dataloader(
        "Data/The Legend of Zelda",
        batch_size=8,
        shuffle=False,
        drop_last=False,
        use_vglc=True,
        room_level=True,
        load_graphs=False,
        dungeon_ids=[9],
        variants=[1, 2],
    )

    assert len(loader.dataset) == 101
    assert {m["dungeon_num"] for m in loader.dataset.sample_metadata} == {9}


def test_build_dungeon_kfold_splits_partitions_cleanly():
    import pytest
    from src.zelda_data.splits import build_dungeon_kfold_splits

    seen_val = set()
    for fold_idx in range(4):
        train_ids, val_ids = build_dungeon_kfold_splits(
            range(1, 9), num_folds=4, fold_index=fold_idx
        )
        assert len(train_ids) == 6
        assert len(val_ids) == 2
        assert set(train_ids).isdisjoint(val_ids)
        assert sorted(set(train_ids) | set(val_ids)) == list(range(1, 9))
        seen_val.update(val_ids)
    assert sorted(seen_val) == list(range(1, 9))

    with pytest.raises(ValueError, match="num_folds"):
        build_dungeon_kfold_splits(range(1, 9), num_folds=1, fold_index=0)
    with pytest.raises(ValueError, match="fold_index"):
        build_dungeon_kfold_splits(range(1, 9), num_folds=4, fold_index=4)


def test_apply_room_symmetry_transform_reflects_node_and_room_positions():
    import torch
    from src.zelda_data.symmetry import apply_room_symmetry_transform

    room = torch.zeros((1, 16, 11), dtype=torch.float32)
    graph_dict = {
        "node_positions": torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 3.0]], dtype=torch.float32),
        "room_position": torch.tensor([1.0, 1.0], dtype=torch.float32),
    }

    _, g_v = apply_room_symmetry_transform(room, graph_dict, flip_h=False, flip_v=True)
    assert torch.allclose(
        g_v["node_positions"],
        torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 3.0]], dtype=torch.float32),
    )
    assert torch.allclose(g_v["room_position"], torch.tensor([0.0, 1.0], dtype=torch.float32))

    _, g_h = apply_room_symmetry_transform(room, graph_dict, flip_h=True, flip_v=False)
    assert torch.allclose(
        g_h["node_positions"],
        torch.tensor([[0.0, 3.0], [1.0, 3.0], [1.0, 1.0]], dtype=torch.float32),
    )
    assert torch.allclose(g_h["room_position"], torch.tensor([1.0, 3.0], dtype=torch.float32))
