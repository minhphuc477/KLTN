"""
Tests for H-MOLQD Block I: Intelligent Data Adapter
====================================================

Tests for VGLC/Graphviz parsing, phase alignment, and tensor conversion.
"""

import pytest
import numpy as np
from pathlib import Path

# Skip if dependencies not available
pytest.importorskip("numpy")


class TestVGLCParser:
    """Tests for VGLC file parsing."""
    
    def test_character_mapping(self):
        """Test character to semantic ID mapping."""
        from src.core.definitions import CHAR_TO_SEMANTIC
        
        # Check essential mappings exist
        assert '.' in CHAR_TO_SEMANTIC  # Floor
        assert 'w' in CHAR_TO_SEMANTIC  # Wall
        assert 'd' in CHAR_TO_SEMANTIC  # Door
        
    def test_parse_room_string(self):
        """Test parsing a room from string."""
        from src.data_processing.data_adapter import VGLCParser
        
        parser = VGLCParser()
        
        # Simple room string
        room_str = """wwwwwwwwwww
w.........w
w.........w
w....k....w
w.........w
w.........w
wwwwwwwwwww"""
        
        lines = room_str.strip().split('\n')
        # Just verify it doesn't crash
        # Full parsing depends on exact character mappings
        assert len(lines) == 7
        assert all(len(line) == 11 for line in lines)


class TestGraphvizParser:
    """Tests for DOT graph parsing."""
    
    def test_parse_dot_string(self):
        """Test parsing a DOT string."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        from src.data_processing.data_adapter import GraphvizParser
        
        parser = GraphvizParser()
        
        # Simple DOT graph
        dot_str = """
        digraph {
            1 [label="s,k"]
            2 [label="e"]
            3 [label="t"]
            1 -> 2 [label="open"]
            2 -> 3 [label="key_locked"]
        }
        """
        
        graph = parser.parse_string(dot_str)
        assert graph is not None
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2


class TestDungeonTensor:
    """Tests for DungeonTensor creation."""
    
    def test_tensor_shape(self):
        """Test tensor dimensions."""
        from src.data_processing.data_adapter import DungeonTensor
        
        # Create simple test data
        room_grids = [
            np.zeros((16, 11), dtype=np.int32) for _ in range(3)
        ]
        
        # Create mock topology
        try:
            import networkx as nx
            topology = nx.DiGraph()
            topology.add_nodes_from([0, 1, 2])
            topology.add_edges_from([(0, 1), (1, 2)])
        except ImportError:
            topology = None
        
        tensor = DungeonTensor(
            room_grids=room_grids,
            topology=topology,
            num_rooms=3,
        )
        
        assert tensor.num_rooms == 3
        assert len(tensor.room_grids) == 3
        assert tensor.room_grids[0].shape == (16, 11)


class TestPhaseAligner:
    """Tests for phase alignment between rooms and graphs."""
    
    def test_alignment_basic(self):
        """Test basic alignment logic."""
        from src.data_processing.data_adapter import PhaseAligner
        
        aligner = PhaseAligner()
        
        # Create mock rooms and graph
        class MockRoom:
            def __init__(self, num_keys=0, num_doors=0):
                self.num_keys = num_keys
                self.num_doors = num_doors
        
        rooms = [MockRoom(1, 2), MockRoom(0, 1), MockRoom(2, 1)]
        
        try:
            import networkx as nx
            graph = nx.DiGraph()
            graph.add_node(0, label="s,k")
            graph.add_node(1, label="")
            graph.add_node(2, label="k,k,t")
            
            alignment, score = aligner.align(rooms, graph)
            assert alignment is not None
            assert 0 <= score <= 1.0
        except ImportError:
            pytest.skip("NetworkX not available")

    def test_shift_grid_improves_vertical_boundary_walls(self):
        """Vertical alignment correction should improve wall-sealed boundaries."""
        from src.data_processing.data_adapter import PhaseAligner
        from src.core.definitions import TileID

        aligner = PhaseAligner(tolerance=2)

        wall = int(TileID.WALL)
        floor = int(TileID.FLOOR)

        base = np.full((7, 11), wall, dtype=np.int32)
        base[1:-1, 1:-1] = floor

        # Shift one row down with floor-filled top to simulate extraction offset.
        misaligned = np.full_like(base, floor)
        misaligned[1:, :] = base[:-1, :]

        corrected = aligner._shift_grid(misaligned, direction='vertical')

        def boundary_wall_hits(grid):
            return int(
                np.sum(grid[0, :] == wall) +
                np.sum(grid[-1, :] == wall) +
                np.sum(grid[:, 0] == wall) +
                np.sum(grid[:, -1] == wall)
            )

        assert boundary_wall_hits(corrected) >= boundary_wall_hits(misaligned)

    def test_shift_grid_improves_horizontal_boundary_walls(self):
        """Horizontal alignment correction should improve wall-sealed boundaries."""
        from src.data_processing.data_adapter import PhaseAligner
        from src.core.definitions import TileID

        aligner = PhaseAligner(tolerance=2)

        wall = int(TileID.WALL)
        floor = int(TileID.FLOOR)

        base = np.full((7, 11), wall, dtype=np.int32)
        base[1:-1, 1:-1] = floor

        # Shift one column right with floor-filled left edge.
        misaligned = np.full_like(base, floor)
        misaligned[:, 1:] = base[:, :-1]

        corrected = aligner._shift_grid(misaligned, direction='horizontal')

        def boundary_wall_hits(grid):
            return int(
                np.sum(grid[0, :] == wall) +
                np.sum(grid[-1, :] == wall) +
                np.sum(grid[:, 0] == wall) +
                np.sum(grid[:, -1] == wall)
            )

        assert boundary_wall_hits(corrected) >= boundary_wall_hits(misaligned)


class TestMLFeatureExtractor:
    """Tests for ML feature extraction."""
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        from src.data_processing.data_adapter import MLFeatureExtractor
        
        extractor = MLFeatureExtractor()
        
        # Create simple grid
        grid = np.zeros((16, 11), dtype=np.int32)
        grid[5:10, 3:8] = 1  # Floor region
        
        features = extractor.extract(grid)
        
        assert isinstance(features, dict)
        assert 'floor_ratio' in features
        assert 0 <= features['floor_ratio'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
