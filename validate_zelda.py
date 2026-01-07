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

Author: KLTN Thesis Project
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

# Import adapter directly from the data folder (renamed to avoid conflict with Data folder)
# We use importlib to handle the path properly
import importlib.util

# Load adapter module
adapter_path = os.path.join(current_dir, 'data', 'adapter.py')
spec = importlib.util.spec_from_file_location("adapter", adapter_path)
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)

# Extract what we need from adapter
IntelligentDataAdapter = adapter_module.IntelligentDataAdapter
SEMANTIC_PALETTE = adapter_module.SEMANTIC_PALETTE
visualize_semantic_grid = adapter_module.visualize_semantic_grid
DungeonData = adapter_module.DungeonData

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
        print("[CALIBRATION] PASSED ✓\n")
        
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
        Validate all processed rooms.
        """
        all_grids = []
        room_info = []
        
        # Collect all grids
        for dungeon_id, dungeon in self.processed_dungeons.items():
            for room_id, room in dungeon.rooms.items():
                all_grids.append(room.grid)
                room_info.append({
                    'dungeon_id': dungeon_id,
                    'room_id': room_id,
                    'contents': room.contents
                })
        
        if not all_grids:
            print("  No rooms to validate!")
            return
        
        # Run batch validation
        batch_result = self.validator.validate_batch(all_grids, verbose=verbose)
        
        # Store results
        self.validation_results['batch'] = batch_result
        self.validation_results['individual'] = []
        
        for i, result in enumerate(batch_result.individual_results):
            self.validation_results['individual'].append({
                **room_info[i],
                'result': result
            })
            
            if result.is_valid_syntax:
                self.stats['valid_rooms'] += 1
            if result.is_solvable:
                self.stats['solvable_rooms'] += 1
        
        # Calculate diversity
        if len(all_grids) >= 2:
            diversity = DiversityEvaluator.batch_diversity(all_grids)
            self.validation_results['diversity'] = diversity
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
        
        print(f"\n📊 STATISTICS")
        print(f"   Total Dungeons:    {self.stats['total_dungeons']}")
        print(f"   Total Rooms:       {self.stats['total_rooms']}")
        print(f"   Valid Syntax:      {self.stats['valid_rooms']}")
        print(f"   Solvable:          {self.stats['solvable_rooms']}")
        
        print(f"\n📈 METRICS")
        if batch:
            print(f"   Solvability Rate:  {100 * batch.solvability_rate:.1f}%")
            print(f"   Avg Reachability:  {100 * batch.avg_reachability:.1f}%")
            print(f"   Avg Path Length:   {batch.avg_path_length:.1f} steps")
            print(f"   Avg Backtracking:  {100 * batch.avg_backtracking:.1f}%")
            print(f"   Diversity Score:   {self.validation_results.get('diversity', 0):.3f}")
        
        print(f"\n⏱️  TIMING")
        print(f"   Processing:        {self.stats['processing_time']:.2f}s")
        print(f"   Validation:        {self.stats['validation_time']:.2f}s")
        print(f"   Total:             {self.stats['processing_time'] + self.stats['validation_time']:.2f}s")
        
        print("\n" + "=" * 60)
        
        # Overall verdict
        if batch and batch.solvability_rate > 0.8:
            print("✅ VALIDATION PASSED: High solvability rate")
        elif batch and batch.solvability_rate > 0.5:
            print("⚠️  VALIDATION WARNING: Moderate solvability rate")
        else:
            print("❌ VALIDATION CONCERNS: Low solvability rate")
        
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
