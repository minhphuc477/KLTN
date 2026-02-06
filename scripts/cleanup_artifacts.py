#!/usr/bin/env python3
"""
Cleanup script for KLTN project artifacts.

Safely removes generated files, caches, and temporary artifacts while
preserving important data like the Zelda dungeon datasets.

Usage:
    python scripts/cleanup_artifacts.py           # Dry-run (list only)
    python scripts/cleanup_artifacts.py --execute # Actually delete
    python scripts/cleanup_artifacts.py --backup  # Move to backups/ instead of delete
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Set

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# Patterns to clean up
PATTERNS_TO_DELETE = {
    # Cache directories
    '__pycache__': 'dir',
    '.pytest_cache': 'dir',
    '.ipynb_checkpoints': 'dir',
    
    # Generated outputs
    'output/processed_data.pkl': 'file',
    'output/validation_results.json': 'file',
    
    # Demo artifacts (safe to regenerate)
    'artifacts/demo': 'dir',
    'artifacts/visual_extracts_demo': 'dir',
    
    # Compiled Python files
    '*.pyc': 'glob',
    '*.pyo': 'glob',
    
    # Editor backups
    '*~': 'glob',
    '*.swp': 'glob',
}

# Directories to recursively clean __pycache__ from
RECURSIVE_CACHE_DIRS = ['src', 'scripts', 'tests', 'Data', 'simulation']

# PROTECTED - never delete these
PROTECTED_PATHS = {
    'Data/The Legend of Zelda',
    'Data/zelda_core.py',
    'Data/__init__.py',
    'src/',
    'tests/',
    'scripts/',
    'gui_runner.py',
    'main.py',
    'requirements*.txt',
    '.git/',
    '.gitignore',
    'README.md',
}


def is_protected(path: Path) -> bool:
    """Check if a path is protected from deletion."""
    rel_path = str(path.relative_to(PROJECT_ROOT))
    
    for protected in PROTECTED_PATHS:
        if protected.endswith('/'):
            # Directory prefix
            if rel_path.startswith(protected) or rel_path == protected.rstrip('/'):
                return True
        elif '*' in protected:
            # Glob pattern
            import fnmatch
            if fnmatch.fnmatch(rel_path, protected):
                return True
        else:
            # Exact match or prefix
            if rel_path == protected or rel_path.startswith(protected + '/'):
                return True
    
    return False


def find_pycache_dirs(root: Path) -> List[Path]:
    """Recursively find all __pycache__ directories."""
    pycache_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        if '__pycache__' in dirnames:
            pycache_dirs.append(Path(dirpath) / '__pycache__')
        # Skip .git and .venv
        dirnames[:] = [d for d in dirnames if d not in {'.git', '.venv', 'venv', 'env'}]
    return pycache_dirs


def find_files_by_glob(root: Path, pattern: str) -> List[Path]:
    """Find files matching a glob pattern."""
    return list(root.rglob(pattern))


def collect_cleanup_targets() -> Tuple[List[Path], List[Path], int]:
    """
    Collect all files/directories that should be cleaned up.
    
    Returns:
        (files_to_delete, dirs_to_delete, total_size_bytes)
    """
    files_to_delete: List[Path] = []
    dirs_to_delete: List[Path] = []
    total_size = 0
    
    # Find all __pycache__ directories recursively
    for cache_dir in find_pycache_dirs(PROJECT_ROOT):
        if not is_protected(cache_dir):
            dirs_to_delete.append(cache_dir)
            total_size += sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    
    # Process patterns
    for pattern, ptype in PATTERNS_TO_DELETE.items():
        if ptype == 'dir':
            target = PROJECT_ROOT / pattern
            if target.exists() and target.is_dir() and not is_protected(target):
                if target not in dirs_to_delete:
                    dirs_to_delete.append(target)
                    total_size += sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
        
        elif ptype == 'file':
            target = PROJECT_ROOT / pattern
            if target.exists() and target.is_file() and not is_protected(target):
                files_to_delete.append(target)
                total_size += target.stat().st_size
        
        elif ptype == 'glob':
            for f in find_files_by_glob(PROJECT_ROOT, pattern):
                if f.is_file() and not is_protected(f):
                    if f not in files_to_delete:
                        files_to_delete.append(f)
                        total_size += f.stat().st_size
    
    return files_to_delete, dirs_to_delete, total_size


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def delete_path(path: Path, backup_dir: Path = None) -> bool:
    """Delete or backup a file/directory."""
    try:
        if backup_dir:
            # Move to backup instead of deleting
            rel_path = path.relative_to(PROJECT_ROOT)
            dest = backup_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
        else:
            # Actually delete
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup KLTN project artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python scripts/cleanup_artifacts.py           # Show what would be deleted
    python scripts/cleanup_artifacts.py --execute # Delete files
    python scripts/cleanup_artifacts.py --backup  # Move to backups/ instead
        '''
    )
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete files (default is dry-run)')
    parser.add_argument('--backup', action='store_true',
                        help='Move files to backups/ instead of deleting')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show more details')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KLTN Project Cleanup Script")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # Collect targets
    print("Scanning for cleanup targets...")
    files, dirs, total_size = collect_cleanup_targets()
    
    if not files and not dirs:
        print("\n✓ No cleanup targets found - project is clean!")
        return 0
    
    # Display what we found
    print(f"\nFound {len(dirs)} directories and {len(files)} files to clean")
    print(f"Total size: {format_size(total_size)}")
    print()
    
    if args.verbose or not args.execute:
        print("Directories:")
        for d in sorted(dirs):
            rel = d.relative_to(PROJECT_ROOT)
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            print(f"  [DIR]  {rel}/ ({format_size(size)})")
        
        print("\nFiles:")
        for f in sorted(files):
            rel = f.relative_to(PROJECT_ROOT)
            size = f.stat().st_size if f.exists() else 0
            print(f"  [FILE] {rel} ({format_size(size)})")
        print()
    
    # Execute or dry-run
    if args.execute or args.backup:
        backup_dir = None
        if args.backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = PROJECT_ROOT / 'backups' / f'cleanup_{timestamp}'
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"Backing up to: {backup_dir}")
        
        action = "Moving" if args.backup else "Deleting"
        print(f"\n{action} files...")
        
        deleted_count = 0
        failed_count = 0
        
        # Delete directories first (they may contain files from our list)
        for d in dirs:
            rel = d.relative_to(PROJECT_ROOT)
            if delete_path(d, backup_dir):
                print(f"  ✓ {rel}/")
                deleted_count += 1
            else:
                failed_count += 1
        
        # Then individual files
        for f in files:
            if f.exists():  # May have been deleted with parent dir
                rel = f.relative_to(PROJECT_ROOT)
                if delete_path(f, backup_dir):
                    print(f"  ✓ {rel}")
                    deleted_count += 1
                else:
                    failed_count += 1
        
        print()
        print(f"{'Moved' if args.backup else 'Deleted'}: {deleted_count} items")
        if failed_count > 0:
            print(f"Failed: {failed_count} items")
            return 1
        
        print("\n✓ Cleanup complete!")
    else:
        print("=" * 60)
        print("DRY RUN - No files were deleted")
        print("Run with --execute to actually delete files")
        print("Run with --backup to move files to backups/ instead")
        print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
