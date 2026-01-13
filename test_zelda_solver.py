"""
ZELDA SOLVER TEST SUITE
=======================
Comprehensive testing of the validator on all VGLC Zelda dungeons.

This script:
1. Loads all 18 dungeons (9 levels x 2 quests)
2. Runs the A* solver on each
3. Reports solvability statistics
4. Identifies logical errors and bugs
5. Tests with multiple personas (speedrunner, balanced, completionist)

Usage:
    python test_zelda_solver.py
    python test_zelda_solver.py --animate  # Show visualization for first solvable level
    python test_zelda_solver.py --persona speedrunner  # Test specific persona

Author: KLTN Thesis Project
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from Data.adapter import IntelligentDataAdapter
from Data.stitcher import DungeonStitcher
from simulation.validator import ZeldaValidator, ValidationResult


class ZeldaSolverTester:
    """Comprehensive tester for Zelda solver across all dungeons."""
    
    def __init__(self, data_root: str):
        """
        Initialize tester.
        
        Args:
            data_root: Path to "The Legend of Zelda" data folder
        """
        self.data_root = Path(data_root)
        self.adapter = IntelligentDataAdapter(str(self.data_root))
        self.validator = ZeldaValidator()
        
        self.results: Dict[str, ValidationResult] = {}
        self.errors: Dict[str, List[str]] = {}
    
    def test_all_dungeons(self, animate_first: bool = False,
                         persona: str = "balanced") -> Dict[str, any]:
        """
        Test solver on all 18 dungeons.
        
        Args:
            animate_first: If True, show animation for first solvable dungeon
            persona: Heuristic mode to use
            
        Returns:
            Dict with summary statistics
        """
        print("="*70)
        print("ZELDA SOLVER TEST SUITE")
        print("="*70)
        print(f"Data Root: {self.data_root}")
        print(f"Persona: {persona}")
        print()
        
        # Process all dungeons
        print("Step 1: Processing dungeons from raw data...")
        dungeons = self.adapter.process_all_dungeons()
        
        if not dungeons:
            print("ERROR: No dungeons were processed!")
            return {}
        
        print(f"Processed {len(dungeons)} dungeons\n")
        
        # Test each dungeon
        print("Step 2: Testing solver on each dungeon...")
        print("-"*70)
        
        solvable_count = 0
        total_count = 0
        animated_one = False
        
        for dungeon_id, dungeon_data in sorted(dungeons.items()):
            total_count += 1
            
            print(f"\nTesting {dungeon_id}:")
            print(f"  Rooms: {len(dungeon_data.rooms)}")
            print(f"  Graph: {dungeon_data.graph.number_of_nodes()} nodes, "
                  f"{dungeon_data.graph.number_of_edges()} edges")
            
            # Stitch dungeon
            stitcher = DungeonStitcher(dungeon_data)
            stitched = stitcher.stitch()
            
            print(f"  Stitched grid: {stitched.global_grid.shape}")
            print(f"  Start: {stitched.start_pos}")
            print(f"  Goal: {stitched.goal_pos}")
            
            # Validate
            result = self.validator.validate_single(
                stitched.global_grid,
                render=(animate_first and not animated_one),
                persona_mode=persona
            )
            
            self.results[dungeon_id] = result
            
            # Report
            if result.is_solvable:
                solvable_count += 1
                print(f"  [OK] SOLVABLE")
                print(f"    Path length: {result.path_length}")
                print(f"    Reachability: {result.reachability:.1%}")
                print(f"    Backtracking: {result.backtracking_score:.2f}")
                
                if animate_first and not animated_one:
                    animated_one = True
                    print("    [Animation shown]")
            else:
                print(f"  [X] NOT SOLVABLE")
                print(f"    Error: {result.error_message}")
            
            if result.logical_errors:
                print(f"  [!] Logical Errors:")
                for err in result.logical_errors[:3]:  # Limit output
                    print(f"    - {err}")
                self.errors[dungeon_id] = result.logical_errors
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Dungeons: {total_count}")
        print(f"Solvable: {solvable_count} ({100*solvable_count/max(1,total_count):.1f}%)")
        print(f"Unsolvable: {total_count - solvable_count}")
        print(f"Dungeons with Logical Errors: {len(self.errors)}")
        
        # Calculate aggregate stats for solvable dungeons
        if solvable_count > 0:
            avg_path = np.mean([r.path_length for r in self.results.values() if r.is_solvable])
            avg_reach = np.mean([r.reachability for r in self.results.values() if r.is_solvable])
            
            print(f"\nAverage Metrics (Solvable Dungeons):")
            print(f"  Path Length: {avg_path:.1f}")
            print(f"  Reachability: {avg_reach:.1%}")
        
        return {
            'total': total_count,
            'solvable': solvable_count,
            'solvability_rate': solvable_count / max(1, total_count),
            'errors': self.errors
        }
    
    def test_multi_persona(self) -> Dict[str, Dict]:
        """
        Test all dungeons with multiple personas.
        
        Returns:
            Dict mapping persona -> results
        """
        print("\n" + "="*70)
        print("MULTI-PERSONA EVALUATION")
        print("="*70)
        
        personas = ["speedrunner", "balanced", "completionist"]
        persona_results = {}
        
        # Process dungeons once
        dungeons = self.adapter.process_all_dungeons()
        
        for persona in personas:
            print(f"\n--- Testing with '{persona}' persona ---\n")
            
            # Create stitched grids
            grids = []
            dungeon_ids = []
            
            for dungeon_id, dungeon_data in sorted(dungeons.items()):
                stitcher = DungeonStitcher(dungeon_data)
                stitched = stitcher.stitch()
                grids.append(stitched.global_grid)
                dungeon_ids.append(dungeon_id)
            
            # Batch validate
            batch_result = self.validator.validate_batch(
                grids,
                verbose=False,
                persona_mode=persona
            )
            
            persona_results[persona] = {
                'solvability': batch_result.solvability_rate,
                'avg_path_length': batch_result.avg_path_length,
                'avg_reachability': batch_result.avg_reachability,
            }
            
            print(f"{persona.capitalize()} Results:")
            print(f"  Solvability: {batch_result.solvability_rate:.1%}")
            print(f"  Avg Path Length: {batch_result.avg_path_length:.1f}")
            print(f"  Avg Reachability: {batch_result.avg_reachability:.1%}")
        
        # Compare personas
        print("\n" + "-"*70)
        print("PERSONA COMPARISON:")
        print("-"*70)
        print(f"{'Persona':<15} {'Solvability':<15} {'Path Length':<15} {'Reachability'}")
        print("-"*70)
        for persona in personas:
            stats = persona_results[persona]
            print(f"{persona:<15} {stats['solvability']:<14.1%} "
                  f"{stats['avg_path_length']:<14.1f} {stats['avg_reachability']:.1%}")
        
        return persona_results
    
    def scan_for_bugs(self):
        """
        Analyze results to identify common bugs and issues.
        """
        print("\n" + "="*70)
        print("BUG SCAN")
        print("="*70)
        
        bugs_found = []
        
        # Check for missing start/goal positions
        missing_start = []
        missing_goal = []
        
        for dungeon_id, result in self.results.items():
            if not result.is_solvable and "No start" in result.error_message:
                missing_start.append(dungeon_id)
            if not result.is_solvable and "No goal" in result.error_message:
                missing_goal.append(dungeon_id)
        
        if missing_start:
            bugs_found.append(f"Missing START position: {len(missing_start)} dungeons")
            print(f"[!] Missing START: {', '.join(missing_start[:5])}")
        
        if missing_goal:
            bugs_found.append(f"Missing GOAL position: {len(missing_goal)} dungeons")
            print(f"[!] Missing GOAL: {', '.join(missing_goal[:5])}")
        
        # Check for graph-text mismatches (ghost rooms)
        for dungeon_id, result in self.results.items():
            if dungeon_id in self.errors:
                for err in self.errors[dungeon_id]:
                    if "ghost" in err.lower() or "missing" in err.lower():
                        bugs_found.append(f"Graph-text mismatch in {dungeon_id}")
                        print(f"[!] {dungeon_id}: {err}")
                        break
        
        if not bugs_found:
            print("[OK] No critical bugs detected!")
        else:
            print(f"\nTotal Issues Found: {len(bugs_found)}")
        
        return bugs_found


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Zelda solver on all dungeons")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Path to 'The Legend of Zelda' data folder")
    parser.add_argument("--animate", action="store_true",
                       help="Show animation for first solvable dungeon")
    parser.add_argument("--persona", type=str, default="balanced",
                       choices=["speedrunner", "balanced", "completionist"],
                       help="Heuristic persona to use")
    parser.add_argument("--multi-persona", action="store_true",
                       help="Test with all personas")
    
    args = parser.parse_args()
    
    # Determine data root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
    
    if not data_root.exists():
        print(f"ERROR: Data folder not found: {data_root}")
        sys.exit(1)
    
    # Run tests
    tester = ZeldaSolverTester(str(data_root))
    
    if args.multi_persona:
        tester.test_multi_persona()
    else:
        tester.test_all_dungeons(
            animate_first=args.animate,
            persona=args.persona
        )
    
    # Scan for bugs
    tester.scan_for_bugs()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
