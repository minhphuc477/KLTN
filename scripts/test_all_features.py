"""
Test All 9 Advanced Features
==============================
Validates that all 9 advanced features are correctly integrated and functional.

Run this before thesis defense to ensure everything works!

Usage:
    python scripts/test_all_features.py --quick      # Fast smoke test
    python scripts/test_all_features.py --thorough   # Full validation
"""

import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    # Feature 1: Seam Smoothing
    from src.generation.seam_smoother import SeamSmoother, SmoothingConfig
    
    # Feature 2: Collision Validator
    from src.validation.collision_alignment_validator import (
        CollisionAlignmentValidator,
        CollisionConfig
    )
    
    # Feature 3: Style Transfer
    from src.generation.style_transfer import ThemeManager, ThemeType
    
    # Feature 4: Fun Metrics
    from src.evaluation.fun_metrics import FunMetricsEvaluator
    
    # Feature 5: Demo Recorder
    from src.utils.demo_recorder import DemoRecorder, RecordingMode, RecordingConfig
    
    # Feature 6: Global State
    from src.generation.global_state import GlobalStateManager, GlobalStateType
    
    # Feature 7: Big Rooms
    from src.generation.big_room_generator import BigRoomGenerator, RoomSize
    
    # Feature 8: LCM-LoRA
    from src.optimization.lcm_lora import LCMLoRAFastSampler, LoRALayer
    
    # Feature 9: Explainability
    from src.utils.explainability import ExplainabilityManager, DecisionTrace, DecisionSource
    from src.utils.explainability_gui import ExplainabilityDebugOverlay
    
    print("✓ All feature imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


class FeatureTester:
    """Comprehensive feature testing."""
    
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = {}
    
    def test_feature_1_seam_smoothing(self):
        """Test seam smoothing functionality."""
        print("\n=== Testing Feature 1: Seam Smoothing ===")
        
        try:
            # Create mock dungeon with visible seams
            dungeon_grid = np.random.randint(0, 3, size=(22, 22))
            
            # Create mock mission graph with adjacency
            mission_graph = nx.Graph()
            mission_graph.add_nodes_from([0, 1])
            mission_graph.add_edge(0, 1)  # Adjacent rooms
            
            # Create layout map (room_id: (x_min, y_min, x_max, y_max))
            layout_map = {
                0: (0, 0, 10, 21),   # Top room
                1: (11, 0, 21, 21)    # Bottom room
            }
            
            # Initialize smoother
            smoother = SeamSmoother(
                config=SmoothingConfig(blend_width=3)
            )
            
            # Smooth seams
            start_time = time.time()
            smoothed = smoother.smooth_dungeon_seams(
                dungeon_grid=dungeon_grid,
                mission_graph=mission_graph,
                layout_map=layout_map
            )
            elapsed = time.time() - start_time
            
            assert smoothed.shape == dungeon_grid.shape
            assert not np.array_equal(smoothed, dungeon_grid)  # Should have changed
            
            print(f"  ✓ Seam smoothing works ({elapsed:.3f}s)")
            print(f"  ✓ Grid shape preserved: {smoothed.shape}")
            
            self.results['seam_smoothing'] = {
                'status': 'PASS',
                'time': elapsed,
                'changes_made': True
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Seam smoothing failed: {e}")
            self.results['seam_smoothing'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_2_collision_validator(self):
        """Test collision alignment validation."""
        print("\n=== Testing Feature 2: Collision Validator ===")
        
        try:
            # Create test room with known structure
            room_grid = np.ones((16, 11), dtype=int)  # All floors
            room_grid[0, :] = 2  # Top wall
            room_grid[-1, :] = 2  # Bottom wall
            room_grid[:, 0] = 2  # Left wall
            room_grid[:, -1] = 2  # Right wall
            room_grid[5, 5] = 2  # Interior wall (creates mismatch if marked walkable)
            
            # Initialize validator
            validator = CollisionAlignmentValidator(
                config=CollisionConfig(alignment_threshold=0.95)
            )
            
            # Validate
            start_time = time.time()
            result = validator.validate_room(
                room_grid=room_grid,
                start_pos=(1, 1),
                goal_pos=(14, 9)
            )
            elapsed = time.time() - start_time
            
            assert result is not None
            assert hasattr(result, 'alignment_score')
            assert 0.0 <= result.alignment_score <= 1.0
            
            print(f"  ✓ Collision validation works ({elapsed:.3f}s)")
            print(f"  ✓ Alignment score: {result.alignment_score:.3f}")
            print(f"  ✓ Valid: {result.is_valid}")
            
            self.results['collision_validator'] = {
                'status': 'PASS',
                'time': elapsed,
                'alignment_score': result.alignment_score
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Collision validator failed: {e}")
            self.results['collision_validator'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_3_style_transfer(self):
        """Test style transfer system."""
        print("\n=== Testing Feature 3: Style Transfer ===")
        
        try:
            # Initialize theme manager
            theme_manager = ThemeManager(themes_dir="assets/themes")
            
            # Set to Zelda theme
            start_time = time.time()
            theme_manager.set_theme("zelda_classic")
            zelda_theme = theme_manager.get_current_theme()
            elapsed = time.time() - start_time
            
            assert zelda_theme is not None
            assert hasattr(zelda_theme, 'theme_name')
            assert hasattr(zelda_theme, 'tile_mappings')
            
            # Test theme application using StyleTransferEngine
            from src.generation.style_transfer import StyleTransferEngine
            engine = StyleTransferEngine()
            
            semantic_grid = np.array([
                [2, 2, 2, 2, 2],
                [2, 1, 1, 1, 2],
                [2, 1, 21, 1, 2],
                [2, 1, 1, 1, 2],
                [2, 2, 2, 2, 2]
            ])
            
            visual_grid = engine.apply_theme(
                semantic_grid=semantic_grid,
                theme_config=zelda_theme
            )
            
            assert visual_grid is not None
            assert visual_grid.shape == (semantic_grid.shape[0], semantic_grid.shape[1], 3)  # RGB
            
            print(f"  ✓ Style transfer works ({elapsed:.3f}s)")
            print(f"  ✓ Theme loaded: {zelda_theme.theme_name}")
            print(f"  ✓ Grid transformed: {semantic_grid.shape} → {visual_grid.shape}")
            
            self.results['style_transfer'] = {
                'status': 'PASS',
                'time': elapsed,
                'theme': zelda_theme.theme_name
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Style transfer failed: {e}")
            self.results['style_transfer'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_4_fun_metrics(self):
        """Test fun metrics evaluation."""
        print("\n=== Testing Feature 4: Fun Metrics ===")
        
        try:
            # Create mock dungeon
            dungeon_grid = np.ones((44, 44), dtype=int)
            
            # Create mock mission graph
            mission_graph = nx.Graph()
            mission_graph.add_nodes_from([0, 1, 2, 3, 4])
            mission_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
            
            # Mock solution path
            solution_path = [0, 1, 2, 3, 4]
            
            # Mock room contents
            room_contents = {
                0: {'enemies': 1, 'keys': 0, 'room_width': 11, 'room_height': 16},
                1: {'enemies': 2, 'keys': 1, 'room_width': 11, 'room_height': 16},
                2: {'enemies': 3, 'keys': 0, 'room_width': 11, 'room_height': 16},
                3: {'enemies': 1, 'keys': 0, 'room_width': 11, 'room_height': 16},
                4: {'enemies': 0, 'keys': 0, 'room_width': 11, 'room_height': 16}  # Goal room
            }
            
            # Mock critical path
            critical_path = set([0, 1, 2, 3, 4])
            
            # Initialize evaluator
            evaluator = FunMetricsEvaluator()
            
            # Evaluate
            start_time = time.time()
            metrics = evaluator.evaluate(
                mission_graph=mission_graph,
                solution_path=solution_path,
                room_contents=room_contents,
                critical_path=critical_path
            )
            elapsed = time.time() - start_time
            
            assert metrics is not None
            assert hasattr(metrics, 'overall_fun_score')
            assert 0.0 <= metrics.overall_fun_score <= 1.0
            assert hasattr(metrics, 'frustration')
            assert hasattr(metrics, 'explorability')
            assert hasattr(metrics, 'flow')
            assert hasattr(metrics, 'pacing')
            
            print(f"  ✓ Fun metrics work ({elapsed:.3f}s)")
            print(f"  ✓ Overall fun score: {metrics.overall_fun_score:.2f}/1.0")
            print(f"  ✓ Frustration: {metrics.frustration.total_frustration:.2f}")
            print(f"  ✓ Explorability: {metrics.explorability.discovery_potential:.2f}")
            
            self.results['fun_metrics'] = {
                'status': 'PASS',
                'time': elapsed,
                'fun_score': metrics.overall_fun_score
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Fun metrics failed: {e}")
            self.results['fun_metrics'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_5_demo_recorder(self):
        """Test demo recording system."""
        print("\n=== Testing Feature 5: Demo Recorder ===")
        
        try:
            # Initialize recorder
            recorder = DemoRecorder(
                config=RecordingConfig(
                    mode=RecordingMode.FAST,
                    output_dir="outputs/test_recordings"
                )
            )
            
            # Start recording
            recorder.start_recording("test_recording")
            
            # Capture test frames
            start_time = time.time()
            for i in range(5):
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                recorder.capture_frame(
                    frame=frame,
                    block=f"Test Block {i}",
                    description=f"Test frame {i}"
                )
            elapsed = time.time() - start_time
            
            # Stop recording
            recorder.stop_recording()
            
            # Verify frames captured
            assert recorder.current_recording is not None
            assert len(recorder.current_recording.frames) == 5
            
            print(f"  ✓ Demo recorder works ({elapsed:.3f}s)")
            print(f"  ✓ Captured {len(recorder.current_recording.frames)} frames")
            
            self.results['demo_recorder'] = {
                'status': 'PASS',
                'time': elapsed,
                'frames_captured': len(recorder.current_recording.frames)
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Demo recorder failed: {e}")
            self.results['demo_recorder'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_6_global_state(self):
        """Test global state system."""
        print("\n=== Testing Feature 6: Global State System ===")
        
        try:
            # Initialize state manager
            manager = GlobalStateManager()
            
            # Add state variable
            start_time = time.time()
            manager.add_state_variable(
                name="water_level",
                state_type=GlobalStateType.WATER_LEVEL,
                initial_value="high"
            )
            
            # Add state transition
            transition = StateTransition(
                from_room=3,
                trigger_condition="switch_pulled",
                state_changes={"water_level": "low"},
                affected_rooms={4, 5, 6}
            )
            manager.add_transition(
                from_room=3,
                trigger_condition="switch_pulled",
                state_changes={"water_level": "low"},
                affected_rooms={4, 5, 6}
            )
            
            # Apply transition
            initial_state = manager.get_state()
            manager.apply_transition(room_id=3, trigger="switch_pulled")
            new_state = manager.get_state()
            elapsed = time.time() - start_time
            
            assert initial_state["water_level"] == "high"
            assert new_state["water_level"] == "low"
            
            print(f"  ✓ Global state works ({elapsed:.3f}s)")
            print(f"  ✓ State change: {initial_state['water_level']} → {new_state['water_level']}")
            print(f"  ✓ Affected rooms: {transition.affected_rooms}")
            
            self.results['global_state'] = {
                'status': 'PASS',
                'time': elapsed,
                'state_transitions': 1
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Global state failed: {e}")
            self.results['global_state'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_7_big_rooms(self):
        """Test big room generation."""
        print("\n=== Testing Feature 7: Big Room Support ===")
        
        try:
            # Test room dimension calculations
            start_time = time.time()
            
            standard = RoomSize.STANDARD
            boss = RoomSize.BOSS
            
            from src.generation.big_room_generator import RoomDimensions
            standard_dims = RoomDimensions.from_size_class(standard)
            boss_dims = RoomDimensions.from_size_class(boss)
            
            elapsed = time.time() - start_time
            
            assert standard_dims.height == 16
            assert standard_dims.width == 11
            assert boss_dims.height == 32
            assert boss_dims.width == 22
            
            print(f"  ✓ Big room support works ({elapsed:.3f}s)")
            print(f"  ✓ Standard room: {standard_dims.height}×{standard_dims.width}")
            print(f"  ✓ Boss room: {boss_dims.height}×{boss_dims.width}")
            
            self.results['big_rooms'] = {
                'status': 'PASS',
                'time': elapsed,
                'max_size': f"{boss_dims.height}×{boss_dims.width}"
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Big room support failed: {e}")
            self.results['big_rooms'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_8_lcm_lora(self):
        """Test LCM-LoRA optimization."""
        print("\n=== Testing Feature 8: LCM-LoRA Performance ===")
        
        try:
            # Test LoRA layer creation
            import torch
            import torch.nn as nn
            
            start_time = time.time()
            
            # Create LoRA layer
            lora_layer = LoRALayer(
                in_features=768,
                out_features=768,
                rank=8,
                alpha=8.0
            )
            
            # Test forward pass
            x = torch.randn(1, 768)
            output = lora_layer(x)
            
            elapsed = time.time() - start_time
            
            assert output.shape == x.shape
            assert lora_layer.rank == 8
            
            # Count parameters
            lora_params = sum(p.numel() for p in lora_layer.parameters())
            full_params = 768 * 768
            reduction = full_params / lora_params
            
            print(f"  ✓ LCM-LoRA works ({elapsed:.3f}s)")
            print(f"  ✓ LoRA params: {lora_params:,} (vs {full_params:,} full)")
            print(f"  ✓ Parameter reduction: {reduction:.1f}×")
            
            self.results['lcm_lora'] = {
                'status': 'PASS',
                'time': elapsed,
                'param_reduction': reduction
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ LCM-LoRA failed: {e}")
            self.results['lcm_lora'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_feature_9_explainability(self):
        """Test explainability system."""
        print("\n=== Testing Feature 9: Explainability System ===")
        
        try:
            from datetime import datetime
            
            # Initialize manager
            manager = ExplainabilityManager(log_dir="outputs/test_explainability")
            
            start_time = time.time()
            
            # Add test traces
            trace1 = DecisionTrace(
                decision_id="test_decision_1",
                source=DecisionSource.GRAMMAR_RULE,
                timestamp=datetime.now(),
                description="Test grammar rule application",
                confidence=0.9,
                affected_elements={"nodes": [0, 1], "edges": [(0, 1)]},
                metadata={"rule_name": "InsertChallenge"}
            )
            manager.add_trace(trace1)
            
            trace2 = DecisionTrace(
                decision_id="test_decision_2",
                source=DecisionSource.DIFFUSION_MODEL,
                timestamp=datetime.now(),
                description="Test diffusion generation",
                confidence=0.85,
                parent_decision="test_decision_1",
                affected_elements={"room": [0]},
                fitness_contribution=0.15
            )
            manager.add_trace(trace2)
            
            # Test queries
            traces_for_node = manager.why_node(0, None)
            all_grammar_traces = manager.get_decisions_by_source(DecisionSource.GRAMMAR_RULE)
            
            elapsed = time.time() - start_time
            
            assert len(manager.traces) == 2
            assert len(traces_for_node) == 1
            assert len(all_grammar_traces) == 1
            
            # Test export
            manager.export_json("outputs/test_explainability/test_traces.json")
            
            print(f"  ✓ Explainability works ({elapsed:.3f}s)")
            print(f"  ✓ Traces stored: {len(manager.traces)}")
            print(f"  ✓ Query successful: found {len(traces_for_node)} traces for node 0")
            print(f"  ✓ JSON export successful")
            
            self.results['explainability'] = {
                'status': 'PASS',
                'time': elapsed,
                'traces_stored': len(manager.traces)
            }
            
            return True
            
        except Exception as e:
            print(f"  ✗ Explainability failed: {e}")
            self.results['explainability'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all feature tests."""
        print("\n" + "="*60)
        print("KLTN 9 Advanced Features - Comprehensive Test Suite")
        print("="*60)
        
        tests = [
            self.test_feature_1_seam_smoothing,
            self.test_feature_2_collision_validator,
            self.test_feature_3_style_transfer,
            self.test_feature_4_fun_metrics,
            self.test_feature_5_demo_recorder,
            self.test_feature_6_global_state,
            self.test_feature_7_big_rooms,
            self.test_feature_8_lcm_lora,
            self.test_feature_9_explainability,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            if test():
                passed += 1
            else:
                failed += 1
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for feature, result in self.results.items():
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_icon} {feature.replace('_', ' ').title()}: {result['status']}")
            if result['status'] == 'PASS' and 'time' in result:
                print(f"   Time: {result['time']:.3f}s")
        
        print(f"\nTotal: {passed}/{passed + failed} tests passed")
        
        if failed == 0:
            print("\n🎉 ALL FEATURES WORKING! Ready for thesis defense! 🎉")
            return 0
        else:
            print(f"\n⚠ {failed} feature(s) need attention")
            return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all 9 advanced features")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--thorough", action="store_true", help="Run thorough validation")
    args = parser.parse_args()
    
    tester = FeatureTester(quick_mode=args.quick)
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
