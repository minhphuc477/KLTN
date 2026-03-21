from types import SimpleNamespace

from src.gui.controls.match_controls import apply_tentative_matches, match_missing_nodes, undo_last_match


class _Logger:
    def info(self, *args, **kwargs):
        return None


class _Room:
    def __init__(self):
        self.graph_node_id = None


class _Matcher:
    def infer_missing_mappings(self, rooms, graph, room_positions=None, room_to_node=None):
        _ = (rooms, graph, room_positions, room_to_node)
        proposed_r2n = {(0, 1): 2, (1, 0): 3}
        proposed_n2r = {2: (0, 1), 3: (1, 0)}
        confidences = {2: 0.9, 3: 0.4}
        return proposed_r2n, proposed_n2r, confidences


def test_match_missing_nodes_applies_confident_and_stages_tentative():
    msgs = []
    current = SimpleNamespace(
        graph=object(),
        rooms={(0, 1): _Room(), (1, 0): _Room()},
        room_positions={(0, 1): (0, 11), (1, 0): (16, 0)},
        room_to_node={},
    )
    gui = SimpleNamespace(
        maps=[current],
        current_map_idx=0,
        widget_manager=SimpleNamespace(widgets=[]),
        match_apply_threshold=0.85,
        match_undo_stack=[],
        _set_message=lambda m, d=0.0: msgs.append((m, d)),
    )

    match_missing_nodes(gui=gui, matcher_cls=_Matcher, logger=_Logger())

    assert current.room_to_node[(0, 1)] == 2
    assert current.rooms[(0, 1)].graph_node_id == 2
    assert (1, 0) not in current.room_to_node
    assert hasattr(current, "match_proposals")
    assert len(gui.match_undo_stack) == 1
    assert "confident matches" in msgs[-1][0]


def test_undo_last_match_restores_snapshot_and_clears_proposals():
    msgs = []
    current = SimpleNamespace(
        rooms={(0, 0): _Room()},
        room_to_node={(0, 0): 4},
        match_proposals=({}, {}, {}),
    )
    gui = SimpleNamespace(
        maps=[current],
        current_map_idx=0,
        match_undo_stack=[{(0, 0): 1}],
        _set_message=lambda m, d=0.0: msgs.append((m, d)),
    )

    undo_last_match(gui=gui, logger=_Logger())

    assert current.room_to_node[(0, 0)] == 1
    assert current.rooms[(0, 0)].graph_node_id == 1
    assert not hasattr(current, "match_proposals")
    assert msgs[-1][0].startswith("Undo: restored")


def test_apply_tentative_matches_applies_above_threshold_and_prunes_staged():
    msgs = []
    current = SimpleNamespace(
        rooms={(0, 0): _Room(), (1, 1): _Room()},
        room_to_node={},
    )
    current.match_proposals = (
        {(0, 0): 10, (1, 1): 11},
        {10: (0, 0), 11: (1, 1)},
        {10: 0.9, 11: 0.5},
    )
    gui = SimpleNamespace(
        maps=[current],
        current_map_idx=0,
        widget_manager=SimpleNamespace(widgets=[]),
        match_apply_threshold=0.85,
        match_undo_stack=[],
        _set_message=lambda m, d=0.0: msgs.append((m, d)),
    )

    apply_tentative_matches(gui=gui, logger=_Logger())

    assert current.room_to_node[(0, 0)] == 10
    assert (1, 1) not in current.room_to_node
    staged_r2n, staged_n2r, _ = current.match_proposals
    assert (1, 1) in staged_r2n
    assert 11 in staged_n2r
    assert len(gui.match_undo_stack) == 1
    assert msgs[-1][0].startswith("Applied 1 tentative matches")
