from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.generation.big_room_generator import BigRoomGenerator, RoomDimensions


class _FakeRoomPipeline:
    def __init__(self):
        self.calls = []

    def generate_room(self, **kwargs):
        self.calls.append(dict(kwargs))
        tile_id = 1 if len(self.calls) % 2 else 43
        return SimpleNamespace(room_grid=np.full((16, 11), tile_id, dtype=np.int32))


def test_big_room_generator_tiles_canonical_rooms_without_mutating_model_shape():
    pipeline = _FakeRoomPipeline()
    generator = BigRoomGenerator(pipeline)

    room = generator.generate_big_room(
        room_id=7,
        dimensions=RoomDimensions.custom(height=22, width=16),
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context={"current_node_idx": 7},
        seed=100,
    )

    assert room.shape == (22, 16)
    assert len(pipeline.calls) == 4
    assert {call["room_id"] for call in pipeline.calls} == {7}
    assert [call["seed"] for call in pipeline.calls] == [100, 101, 102, 103]
    assert set(np.unique(room)).issubset({1, 43})


def test_big_room_single_patch_crops_canonical_output():
    pipeline = _FakeRoomPipeline()
    generator = BigRoomGenerator(pipeline)

    room = generator.generate_big_room(
        room_id=3,
        dimensions=RoomDimensions.custom(height=8, width=9),
        neighbor_latents={},
        graph_context={},
    )

    assert room.shape == (8, 9)
    assert len(pipeline.calls) == 1
