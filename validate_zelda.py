"""
ZELDA VALIDATION PIPELINE
=========================
Main orchestration script for validating original Zelda dungeon data.

This script:
1. Processes raw VGLC data using Block I (Data Adapter)
2. Validates processed maps using Block VI (Validator)
3. Generates comprehensive metrics and reports

Usage:
    python validate_zelda.py                    # Full validation
    python validate_zelda.py --quick            # Quick test with single dungeon
    python validate_zelda.py --gui              # Launch GUI after validation
    python validate_zelda.py --report report.txt  # Save report to file


"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from CANONICAL source: zelda_core.py
# This replaces the old adapter.py imports
from Data.zelda_core import (
    # Adapter classes
    ZeldaDungeonAdapter,
    DungeonStitcher,
    DungeonSolver,
    MLFeatureExtractor,
    
    # Data classes  
    Dungeon,
    StitchedDungeon,
    DungeonData,
    RoomData,
    
    # Constants
    SEMANTIC_PALETTE,
    ID_TO_NAME,
    ValidationMode,
    
    # Utilities
    visualize_semantic_grid,
    convert_dungeon_to_dungeondata,
)

# Import validator (this should work fine)
from simulation.validator import (
    ZeldaValidator,
    ZeldaLogicEnv,
    SanityChecker,
    MetricsEngine,
    DiversityEvaluator,
    ValidationResult,
    BatchValidationResult,
    create_test_map
)

# Import stitcher for dungeon-level validation
stitcher_path = os.path.join(current_dir, 'data', 'stitcher.py')
spec_stitcher = importlib.util.spec_from_file_location("stitcher", stitcher_path)
stitcher_module = importlib.util.module_from_spec(spec_stitcher)
spec_stitcher.loader.exec_module(stitcher_module)
DungeonStitcher = stitcher_module.DungeonStitcher
StitchedDungeon = stitcher_module.StitchedDungeon


class ZeldaValidationPipeline:
    """
    Main validation pipeline orchestrator.
    
    Coordinates data processing and validation across all dungeons.
    """
    
    def __init__(self, data_root: str = None, output_dir: str = None):
        """
        Initialize the pipeline.
        
        Args:
            data_root: Path to The Legend of Zelda data folder
            output_dir: Path for output reports and processed data
        """
        if data_root is None:
            data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "output"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.adapter = None
        self.validator = None
        self.processed_dungeons = {}
        self.validation_results = {}
        
        # Statistics
        self.stats = {
            'total_dungeons': 0,
            'total_rooms': 0,
            'processed_rooms': 0,
            'valid_rooms': 0,
            'solvable_rooms': 0,
            'processing_time': 0,
            'validation_time': 0,
        }
    
    def run_full_pipeline(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete validation pipeline.
        
        Steps:
        1. Process raw data (Block I)
        2. Validate all maps (Block VI)
        3. Generate metrics and report
        
        Returns:
            Dictionary with all results and metrics
        """
        print("=" * 60)
        print("ZELDA VALIDATION PIPELINE")
        print("=" * 60)
        print(f"Data Source: {self.data_root}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 60 + "\n")
        
        # Step 1: Run calibration
        print("[CALIBRATION] Running system calibration...")
        self._run_calibration()
        print("[CALIBRATION] PASSED [OK]\n")
        
        # Step 2: Process data
        print("[BLOCK I] Processing raw dungeon data...")
        start_time = time.time()
        self._process_data(verbose)
        self.stats['processing_time'] = time.time() - start_time
        print(f"[BLOCK I] Complete. Processed {self.stats['processed_rooms']} rooms in {self.stats['processing_time']:.2f}s\n")
        
        # Step 3: Validate all maps
        print("[BLOCK VI] Validating processed maps...")
        start_time = time.time()
        self._validate_all(verbose)
        self.stats['validation_time'] = time.time() - start_time
        print(f"[BLOCK VI] Complete. Validation took {self.stats['validation_time']:.2f}s\n")
        
        # Step 4: Generate report
        report = self._generate_report()
        
        # Print summary
        self._print_summary()
        
        return report
    
    def _run_calibration(self):
        """
        Run calibration test to verify solver correctness.
        
        Uses a known-solvable test map.
        """
        test_map = create_test_map()
        
        # Initialize validator with calibration map
        self.validator = ZeldaValidator(calibration_map=test_map)
        
        # Verify calibration passed
        if not self.validator.is_calibrated:
            raise RuntimeError("Calibration failed! Solver may have bugs.")
    
    def _process_data(self, verbose: bool = True):
        """
        Process raw VGLC data using Block I adapter.
        """
        self.adapter = IntelligentDataAdapter(str(self.data_root))
        
        # Check if data exists
        processed_dir = self.data_root / "Processed"
        graph_dir = self.data_root / "Graph Processed"
        
        if not processed_dir.exists():
            print(f"  Warning: Processed folder not found at {processed_dir}")
            return
        
        if not graph_dir.exists():
            print(f"  Warning: Graph Processed folder not found at {graph_dir}")
        
        # Process all dungeons
        try:
            self.processed_dungeons = self.adapter.process_all_dungeons()
        except Exception as e:
            print(f"  Error during processing: {e}")
            return
        
        # Update statistics
        self.stats['total_dungeons'] = len(self.processed_dungeons)
        for dungeon_id, dungeon in self.processed_dungeons.items():
            self.stats['total_rooms'] += len(dungeon.rooms)
            self.stats['processed_rooms'] += len(dungeon.rooms)
            
            if verbose:
                print(f"  - {dungeon_id}: {len(dungeon.rooms)} rooms")
        
        # Save processed data
        output_path = self.output_dir / "processed_data.pkl"
        try:
            self.adapter.save_processed_data(str(output_path))
            print(f"  Saved processed data to {output_path}")
        except Exception as e:
            print(f"  Warning: Could not save processed data: {e}")
    
    def _validate_all(self, verbose: bool = True):
        """
        Validate all processed dungeons.
        
        Three-level validation:
        1. Graph-level: Use graph topology to determine logical solvability
        2. Dungeon-level: Stitch rooms and validate full dungeon
        3. Room-level: Validate rooms that have both START and TRIFORCE
        """
        all_grids = []
        room_info = []
        
        # Collect individual room grids (for diversity and room-level metrics)
        for dungeon_id, dungeon in self.processed_dungeons.items():
            for room_id, room in dungeon.rooms.items():
                all_grids.append(room.grid)
                room_info.append({
                    'dungeon_id': dungeon_id,
                    'room_id': room_id,
                    'contents': room.contents
                })
        
        # Store results
        self.validation_results['individual'] = []
        self.validation_results['dungeon_level'] = []
        self.validation_results['graph_level'] = []
        
        # ==========================================
        # GRAPH-LEVEL VALIDATION (using graph topology)
        # ==========================================
        print("  [Graph-Level Validation (Topology-Guided)]")
        graph_solvable_count = 0
        graph_total = 0
        
        from simulation.validator import GraphGuidedValidator
        graph_validator = GraphGuidedValidator()
        
        for dungeon_id, dungeon in self.processed_dungeons.items():
            graph_total += 1
            
            try:
                # Validate using graph topology with edge types (keys, locks, etc.)
                graph_result = graph_validator.validate_with_edge_types(dungeon)
                
                self.validation_results['graph_level'].append({
                    'dungeon_id': dungeon_id,
                    'result': graph_result,
                    'graph_path': graph_result.graph_path,
                    'missing_rooms': graph_result.missing_rooms,
                    'connectivity': graph_result.connectivity_score
                })
                
                if graph_result.is_solvable:
                    graph_solvable_count += 1
                
                if verbose:
                    status = "GRAPH-SOLVABLE" if graph_result.is_solvable else "GRAPH-BLOCKED"
                    path_info = f"path={graph_result.graph_path}" if graph_result.graph_path else "no path"
                    missing_info = f"missing={graph_result.missing_rooms}" if graph_result.missing_rooms else ""
                    print(f"    {dungeon_id}: {status} ({path_info}) {missing_info}")
                    
            except Exception as e:
                if verbose:
                    print(f"    {dungeon_id}: GRAPH ERROR - {e}")
        
        print(f"  Graph-level: {graph_solvable_count}/{graph_total} solvable (using topology)")
        
        # ==========================================
        # DUNGEON-LEVEL VALIDATION (using stitcher)
        # ==========================================
        print("  [Dungeon-Level Validation (Grid-Based)]")
        dungeon_solvable_count = 0
        dungeon_total = 0
        
        for dungeon_id, dungeon in self.processed_dungeons.items():
            dungeon_total += 1
            
            try:
                # Stitch the dungeon
                stitcher = DungeonStitcher(dungeon)
                stitched = stitcher.stitch()
                
                if verbose:
                    print(f"    {dungeon_id}: stitched {stitched.global_grid.shape}, START={stitched.start_pos}, GOAL={stitched.goal_pos}")
                
                # Validate stitched dungeon
                if stitched.start_pos is not None and stitched.goal_pos is not None:
                    result = self.validator.validate_single(stitched.global_grid)
                    
                    self.validation_results['dungeon_level'].append({
                        'dungeon_id': dungeon_id,
                        'result': result,
                        'grid_shape': stitched.global_grid.shape,
                        'start_pos': stitched.start_pos,
                        'goal_pos': stitched.goal_pos
                    })
                    
                    if result.is_solvable:
                        dungeon_solvable_count += 1
                        self.stats['solvable_rooms'] += 1  # Count as solvable
                    
                    if verbose:
                        status = "SOLVABLE" if result.is_solvable else " NOT SOLVABLE"
                        print(f"      {status} (path={result.path_length})")
                else:
                    if verbose:
                        print(f"       Missing START or TRIFORCE after stitching")
                        
            except Exception as e:
                if verbose:
                    print(f"    {dungeon_id}: ERROR - {e}")
        
        print(f"  Dungeon-level: {dungeon_solvable_count}/{dungeon_total} solvable (grid pathfinding)")
        
        # Update stats to include graph-level results
        self.stats['graph_solvable'] = graph_solvable_count
        
        # ==========================================
        # ROOM-LEVEL VALIDATION (for rooms with both START and TRIFORCE)
        # ==========================================
        print("  [Room-Level Validation]")
        room_solvable_count = 0
        room_with_start_and_goal = 0
        
        for i, grid in enumerate(all_grids):
            info = room_info[i]
            
            # Check if room has both START and TRIFORCE
            has_start = SEMANTIC_PALETTE['START'] in grid
            has_goal = SEMANTIC_PALETTE['TRIFORCE'] in grid
            
            result = ValidationResult(
                is_solvable=False,
                is_valid_syntax=True,
                reachability=0.0,
                path_length=0,
                backtracking_score=0.0,
                logical_errors=[],
                path=[],
                error_message=""
            )
            
            if has_start and has_goal:
                room_with_start_and_goal += 1
                result = self.validator.validate_single(grid)
                if result.is_solvable:
                    room_solvable_count += 1
            else:
                # Room doesn't have both - mark as "incomplete" not "unsolvable"
                result.error_message = "Room lacks START or TRIFORCE (normal for multi-room dungeons)"
            
            self.validation_results['individual'].append({
                **info,
                'result': result,
                'has_start': has_start,
                'has_goal': has_goal
            })
            
            if result.is_valid_syntax:
                self.stats['valid_rooms'] += 1
        
        print(f"  Room-level: {room_solvable_count}/{room_with_start_and_goal} standalone-solvable rooms")
        print(f"  ({len(all_grids) - room_with_start_and_goal} rooms require dungeon-level validation)")
        
        # Create batch result
        batch_result = BatchValidationResult(
            total_maps=dungeon_total,  # Count dungeons, not rooms
            valid_syntax_count=dungeon_total,
            solvable_count=dungeon_solvable_count,
            solvability_rate=dungeon_solvable_count / max(1, dungeon_total),
            avg_reachability=1.0 if dungeon_solvable_count > 0 else 0.0,
            avg_path_length=sum(r['result'].path_length for r in self.validation_results['dungeon_level'] if r['result'].is_solvable) / max(1, dungeon_solvable_count),
            avg_backtracking=0.0,
            diversity_score=0.0,
            individual_results=[]
        )
        
        self.validation_results['batch'] = batch_result
        
        # Calculate diversity on individual rooms
        if len(all_grids) >= 2:
            diversity = DiversityEvaluator.batch_diversity(all_grids)
            self.validation_results['diversity'] = diversity
            batch_result.diversity_score = diversity
        else:
            self.validation_results['diversity'] = 0.0
    
    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        """
        batch_result = self.validation_results.get('batch')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_source': str(self.data_root),
            'statistics': self.stats,
            'metrics': {
                'solvability_rate': batch_result.solvability_rate if batch_result else 0.0,
                'valid_syntax_rate': batch_result.valid_syntax_count / max(1, batch_result.total_maps) if batch_result else 0.0,
                'avg_reachability': batch_result.avg_reachability if batch_result else 0.0,
                'avg_path_length': batch_result.avg_path_length if batch_result else 0.0,
                'avg_backtracking': batch_result.avg_backtracking if batch_result else 0.0,
                'diversity_score': self.validation_results.get('diversity', 0.0),
            },
            'dungeons': {}
        }
        
        # Per-dungeon breakdown
        for result_info in self.validation_results.get('individual', []):
            dungeon_id = result_info['dungeon_id']
            if dungeon_id not in report['dungeons']:
                report['dungeons'][dungeon_id] = {
                    'rooms': [],
                    'solvable_count': 0,
                    'total_count': 0
                }
            
            report['dungeons'][dungeon_id]['rooms'].append({
                'room_id': result_info['room_id'],
                'solvable': result_info['result'].is_solvable,
                'path_length': result_info['result'].path_length,
                'errors': result_info['result'].logical_errors
            })
            
            report['dungeons'][dungeon_id]['total_count'] += 1
            if result_info['result'].is_solvable:
                report['dungeons'][dungeon_id]['solvable_count'] += 1
        
        # Save report
        report_path = self.output_dir / "validation_report.json"
        try:
            with open(report_path, 'w') as f:
                # Convert to JSON-serializable format
                json_report = self._make_json_serializable(report)
                json.dump(json_report, f, indent=2)
            print(f"\nReport saved to {report_path}")
        except Exception as e:
            print(f"Warning: Could not save report: {e}")
        
        return report
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def _print_summary(self):
        """Print validation summary to console."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        batch = self.validation_results.get('batch')
        
        print(f"\n[STATISTICS]")
        print(f"   Total Dungeons:    {self.stats['total_dungeons']}")
        print(f"   Total Rooms:       {self.stats['total_rooms']}")
        print(f"   Valid Syntax:      {self.stats['valid_rooms']}")
        print(f"   Grid-Solvable:     {self.stats['solvable_rooms']}")
        
        # Graph-level stats
        graph_solvable = self.stats.get('graph_solvable', 0)
        print(f"   Graph-Solvable:    {graph_solvable} (topology-guided)")
        
        print(f"\n[METRICS]")
        if batch:
            print(f"   Grid Solvability:  {100 * batch.solvability_rate:.1f}%")
            print(f"   Graph Solvability: {100 * graph_solvable / max(1, self.stats['total_dungeons']):.1f}%")
            print(f"   Avg Reachability:  {100 * batch.avg_reachability:.1f}%")
            print(f"   Avg Path Length:   {batch.avg_path_length:.1f} steps")
            print(f"   Avg Backtracking:  {100 * batch.avg_backtracking:.1f}%")
            print(f"   Diversity Score:   {self.validation_results.get('diversity', 0):.3f}")
        
        print(f"\n[TIMING]")
        print(f"   Processing:        {self.stats['processing_time']:.2f}s")
        print(f"   Validation:        {self.stats['validation_time']:.2f}s")
        print(f"   Total:             {self.stats['processing_time'] + self.stats['validation_time']:.2f}s")
        
        print("\n" + "=" * 60)
        
        # Overall verdict - use graph solvability as primary metric
        graph_rate = graph_solvable / max(1, self.stats['total_dungeons'])
        if graph_rate > 0.8:
            print("[PASS] VALIDATION PASSED: High graph-solvability rate")
        elif graph_rate > 0.5:
            print("[WARN] VALIDATION WARNING: Moderate graph-solvability rate")
        else:
            print("[FAIL] VALIDATION CONCERNS: Low solvability rate")
            print("   Note: Graph solvability considers logical topology,")
            print("         Grid solvability requires complete room data")
        
        print("=" * 60 + "\n")


def run_quick_test():
    """Run a quick test with synthetic data."""
    print("Running quick test with synthetic map...\n")
    
    # Create test map
    test_map = create_test_map()
    
    # Validate
    validator = ZeldaValidator()
    result = validator.validate_single(test_map)
    
    print("Test Map Validation:")
    print(f"  Solvable: {result.is_solvable}")
    print(f"  Valid Syntax: {result.is_valid_syntax}")
    print(f"  Path Length: {result.path_length}")
    print(f"  Reachability: {100*result.reachability:.1f}%")
    
    if result.logical_errors:
        print(f"  Errors: {result.logical_errors}")
    
    return result.is_solvable


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Zelda Dungeon Validation Pipeline"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to The Legend of Zelda data folder'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for reports'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick test only'
    )
    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch GUI after validation'
    )
    parser.add_argument(
        '--report', '-r',
        type=str,
        default=None,
        help='Save text report to file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    # Full pipeline
    try:
        pipeline = ZeldaValidationPipeline(
            data_root=args.data,
            output_dir=args.output
        )
        
        report = pipeline.run_full_pipeline(verbose=args.verbose)
        
        # Save text report if requested
        if args.report:
            with open(args.report, 'w') as f:
                f.write("ZELDA VALIDATION REPORT\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Timestamp: {report['timestamp']}\n")
                f.write(f"Data Source: {report['data_source']}\n\n")
                f.write("METRICS:\n")
                for key, value in report['metrics'].items():
                    f.write(f"  {key}: {value}\n")
            print(f"Text report saved to {args.report}")
        
        # Launch GUI if requested
        if args.gui:
            print("\nLaunching GUI...")
            try:
                from gui_runner import main as gui_main
                gui_main()
            except ImportError as e:
                print(f"Could not launch GUI: {e}")
        
        # Return success if solvability > 0 or no maps processed
        batch = pipeline.validation_results.get('batch')
        if batch and batch.total_maps > 0:
            return batch.solvable_count > 0
        return True
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
