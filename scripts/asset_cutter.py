"""
ASSET CUTTER UTILITY
====================
Slices the Zelda dungeon tileset image into individual 16x16 tile images
for use in GUI visualization and debugging.

This eliminates the need for manual Photoshop work and ensures consistent
tile extraction from the source tileset.

Usage:
    python scripts/asset_cutter.py

Output:
    Creates assets/raw/ folder with numbered tiles (tile_Y_X.png)
    User must manually select the best tiles and rename them to:
    - floor.png
    - wall.png
    - door_open.png
    - door_locked.png
    - key.png
    - link.png
    - enemy.png
    - etc.

"""

import os
import sys
from pathlib import Path
from PIL import Image


class TilesetCutter:
    """Cuts a tileset image into individual tiles."""
    
    def __init__(self, tile_size: int = 16, margin: int = 0, padding: int = 0):
        """
        Initialize the cutter.
        
        Args:
            tile_size: Size of each tile in pixels (NES standard is 16x16)
            margin: Border around the entire tileset in pixels
            padding: Space between tiles in pixels
        """
        self.tile_size = tile_size
        self.margin = margin
        self.padding = padding
    
    def cut(self, input_path: str, output_dir: str, filter_empty: bool = True) -> int:
        """
        Slice tileset into individual tiles.
        
        Args:
            input_path: Path to source tileset image
            output_dir: Directory to save individual tiles
            filter_empty: If True, skip tiles that are solid color or empty
            
        Returns:
            Number of tiles extracted
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Tileset not found: {input_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tileset
        img = Image.open(input_path)
        
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        width, height = img.size
        
        print(f"Processing tileset: {width}x{height} pixels")
        print(f"Tile size: {self.tile_size}x{self.tile_size}")
        print(f"Margin: {self.margin}px, Padding: {self.padding}px")
        
        tile_count = 0
        skipped_count = 0
        
        # Calculate starting position accounting for margin
        start_x = self.margin
        start_y = self.margin
        
        # Calculate stride (tile size + padding)
        stride = self.tile_size + self.padding
        
        # Iterate through grid positions
        y = start_y
        row = 0
        
        while y + self.tile_size <= height:
            x = start_x
            col = 0
            
            while x + self.tile_size <= width:
                # Extract tile
                box = (x, y, x + self.tile_size, y + self.tile_size)
                tile = img.crop(box)
                
                # Check if tile should be saved
                if filter_empty and self._is_empty_tile(tile):
                    skipped_count += 1
                else:
                    # Save tile with row/col in filename for easy identification
                    filename = f"tile_r{row:02d}_c{col:02d}.png"
                    tile_path = os.path.join(output_dir, filename)
                    tile.save(tile_path)
                    tile_count += 1
                
                x += stride
                col += 1
            
            y += stride
            row += 1
        
        print(f"\nExtraction complete!")
        print(f"  - Tiles saved: {tile_count}")
        print(f"  - Tiles skipped (empty): {skipped_count}")
        print(f"  - Output directory: {output_dir}")
        
        return tile_count
    
    def _is_empty_tile(self, tile: Image.Image) -> bool:
        """
        Check if a tile is empty or solid color (likely unused).
        
        Uses variance heuristic: if all pixels are nearly identical, skip.
        """
        # Convert to numpy for analysis
        import numpy as np
        
        arr = np.array(tile)
        
        # Check alpha channel first - if all transparent, it's empty
        if arr.shape[2] == 4:  # RGBA
            alpha = arr[:, :, 3]
            if np.all(alpha == 0):
                return True
        
        # Check variance in RGB channels
        # Low variance = solid color = probably not useful
        rgb = arr[:, :, :3]
        variance = np.var(rgb)
        
        # If variance is very low, tile is likely solid color
        return variance < 10.0


def main():
    """Main entry point for the asset cutter."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Input: Look for the dungeon tileset
    # Common locations:
    # - Data/Assets/NES - The Legend of Zelda - Tilesets - Dungeon Tileset.png
    # - assets/Dungeon Tileset.png
    
    possible_inputs = [
        project_root / "Data" / "Assets" / "NES - The Legend of Zelda - Tilesets - Dungeon Tileset.png",
        project_root / "Data" / "Assets" / "Dungeon Tileset.png",
        project_root / "assets" / "Dungeon Tileset.png",
        project_root / "Data" / "Assets" / "tileset.png",
    ]
    
    input_path = None
    for path in possible_inputs:
        if path.exists():
            input_path = str(path)
            break
    
    if input_path is None:
        print("ERROR: Could not find dungeon tileset image!")
        print("Expected one of:")
        for p in possible_inputs:
            print(f"  - {p}")
        print("\nPlease place the tileset image in one of these locations.")
        sys.exit(1)
    
    # Output directory
    output_dir = project_root / "assets" / "raw"
    
    print("="*60)
    print("ZELDA TILESET CUTTER")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Create cutter and process
    cutter = TilesetCutter(tile_size=16, margin=1, padding=1)
    
    try:
        tile_count = cutter.cut(input_path, str(output_dir), filter_empty=True)
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print(f"1. Open the folder: {output_dir}")
        print("2. Browse the extracted tiles")
        print("3. Identify the best tiles for:")
        print("   - Floor (walkable ground)")
        print("   - Wall (solid barrier)")
        print("   - Door Open (passage)")
        print("   - Door Locked (key door)")
        print("   - Key (small key icon)")
        print("   - Link (player sprite)")
        print("   - Enemy (monster sprite)")
        print("   - Triforce (goal)")
        print("4. Copy selected tiles to assets/ and rename them")
        print("   Example: tile_r03_c05.png -> floor.png")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
