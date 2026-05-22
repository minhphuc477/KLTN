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

    assert len(train) == 356
    assert len(test) == 103

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

    assert len(loader.dataset) == 103
    assert {m["dungeon_num"] for m in loader.dataset.sample_metadata} == {9}
