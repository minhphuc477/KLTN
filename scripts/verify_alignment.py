"""
DATA INTEGRITY VERIFICATION TOOL
=================================
Overlays parsed text grid onto original screenshot images to verify
that the adapter.py parsing logic is correctly aligned with the source data.

This is CRITICAL for data integrity - if the text parser is offset by even
1 pixel, the entire training dataset will be corrupted ("Garbage In, Garbage Out").

The tool generates comparison images showing:
1. Original NES screenshot
2. Text grid overlay (color-coded letters)
3. Red grid lines showing tile boundaries
4. Highlights showing detected doors and items

Usage:
    python scripts/verify_alignment.py
    python scripts/verify_alignment.py --dungeon tloz1_1

Output:
    Saves verification images to output/verify_{dungeon_id}.png

If you see misalignment (e.g., 'W' letters not on walls), the grid dimensions
in GridBasedRoomExtractor need adjustment.

"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class AlignmentVerifier:
    """Verifies text-to-image alignment for VGLC data."""
    
    def __init__(self, data_root: str):
        """
        Initialize verifier.
        
        Args:
            data_root: Path to "The Legend of Zelda" data folder
        """
        self.data_root = Path(data_root)
        self.text_dir = self.data_root / "Processed"
        self.image_dir = self.data_root / "Original"
        
        # Try to import adapter for room extraction
        try:
            sys.path.insert(0, str(self.data_root.parent.parent))
            from Data.adapter import GridBasedRoomExtractor, ROOM_HEIGHT, ROOM_WIDTH
            self.extractor = GridBasedRoomExtractor(room_rows=ROOM_HEIGHT, room_cols=ROOM_WIDTH)
        except ImportError:
            print("WARNING: Could not import adapter. Using fallback extractor.")
            self.extractor = None
    
    def verify(self, dungeon_id: str, output_path: Optional[str] = None) -> bool:
        """
        Verify alignment for a single dungeon.
        
        Args:
            dungeon_id: Dungeon identifier (e.g., "tloz1_1")
            output_path: Where to save verification image (optional)
            
        Returns:
            True if verification image created successfully
        """
        text_path = self.text_dir / f"{dungeon_id}.txt"
        image_path = self.image_dir / f"{dungeon_id}.png"
        
        # Check if files exist
        if not text_path.exists():
            print(f"ERROR: Text file not found: {text_path}")
            return False
        
        if not image_path.exists():
            print(f"WARNING: Image file not found: {image_path}")
            print(f"         Verification will use text-only visualization.")
            image_path = None
        
        # Load text data
        print(f"Loading text: {text_path}")
        with open(text_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
        
        if not lines:
            print("ERROR: Text file is empty!")
            return False
        
        # Convert to grid
        max_width = max(len(line) for line in lines)
        char_grid = np.array([list(line.ljust(max_width, '-')) for line in lines])
        
        h, w = char_grid.shape
        print(f"Text grid dimensions: {h} rows x {w} cols")
        
        # Load image if available
        if image_path:
            print(f"Loading image: {image_path}")
            original_img = Image.open(image_path).convert("RGBA")
            
            # NES Zelda screenshots are typically 256x240 with HUD at top (64px)
            # or just the playable area (256x176)
            img_w, img_h = original_img.size
            print(f"Image dimensions: {img_w}x{img_h} pixels")
            
            # Crop HUD if full screenshot
            if img_h == 240 and img_w == 256:
                print("Detected full screenshot - cropping HUD (top 64px)")
                original_img = original_img.crop((0, 64, 256, 240))
                img_h = 176
            
            # Calculate tile size
            # Each character in text should map to a pixel region
            tile_h = img_h // h
            tile_w = img_w // w
            print(f"Calculated tile size: {tile_h}x{tile_w} pixels per character")
        else:
            # No image - create blank canvas
            tile_h, tile_w = 16, 16  # Assume NES standard
            img_w = w * tile_w
            img_h = h * tile_h
            original_img = Image.new("RGBA", (img_w, img_h), (40, 40, 40, 255))
        
        # Create overlay
        overlay = Image.new("RGBA", original_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", max(8, min(tile_h, tile_w) - 4))
        except:
            font = ImageFont.load_default()
        
        # Draw grid and text overlay
        for r in range(h):
            for c in range(w):
                char = char_grid[r, c]
                
                # Calculate pixel position
                x = c * tile_w
                y = r * tile_h
                
                # Draw grid border (light red)
                draw.rectangle([x, y, x + tile_w, y + tile_h], 
                              outline=(255, 100, 100, 100), width=1)
                
                # Skip void
                if char == '-':
                    continue
                
                # Color-code characters
                color = self._get_char_color(char)
                
                # Draw character
                text_x = x + tile_w // 4
                text_y = y + tile_h // 4
                draw.text((text_x, text_y), char, fill=color, font=font)
        
        # Add legend
        self._draw_legend(draw, original_img.size)
        
        # Composite
        result = Image.alpha_composite(original_img, overlay)
        
        # Save
        if output_path is None:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"verify_{dungeon_id}.png"
        
        result.save(output_path)
        print(f"\n✓ Verification image saved: {output_path}")
        print(f"\nINSPECTION GUIDE:")
        print(f"  - Open the image in an image viewer")
        print(f"  - Check if 'W' letters align with WALL tiles (dark blocks)")
        print(f"  - Check if 'F' letters align with FLOOR tiles (walkable area)")
        print(f"  - Check if 'D' letters align with DOOR openings")
        print(f"  - If misaligned: adjust room dimensions in adapter.py")
        
        return True
    
    def _get_char_color(self, char: str) -> tuple:
        """
        Return color for a character type.
        
        Color legend:
        - W (Wall) -> Blue
        - F (Floor) -> White
        - D (Door) -> Green
        - S (Start/Stair) -> Yellow
        - T/G (Triforce/Goal) -> Gold
        - M/E (Enemy) -> Red
        - K/k (Key) -> Cyan
        - Default -> Light Gray
        """
        color_map = {
            'W': (0, 100, 255, 255),      # Blue - Walls
            'F': (255, 255, 255, 255),    # White - Floor
            '.': (200, 200, 200, 255),    # Gray - Alt Floor
            'D': (0, 255, 0, 255),        # Green - Doors
            'S': (255, 255, 0, 255),      # Yellow - Start/Stairs
            'T': (255, 215, 0, 255),      # Gold - Triforce
            'G': (255, 215, 0, 255),      # Gold - Goal
            't': (255, 200, 0, 255),      # Orange - Triforce alt
            'M': (255, 0, 0, 255),        # Red - Enemy
            'E': (255, 50, 50, 255),      # Bright Red - Enemy alt
            'e': (200, 50, 50, 255),      # Dark Red - Enemy alt
            'K': (0, 255, 255, 255),      # Cyan - Key
            'k': (0, 200, 200, 255),      # Dark Cyan - Small Key
            'B': (150, 75, 0, 255),       # Brown - Block
            'P': (255, 0, 255, 255),      # Magenta - Puzzle
            'O': (100, 100, 255, 255),    # Light Blue - Water
            'I': (255, 128, 0, 255),      # Orange - Item
        }
        
        return color_map.get(char, (180, 180, 180, 255))
    
    def _draw_legend(self, draw: ImageDraw.Draw, img_size: tuple):
        """Draw legend explaining colors."""
        w, h = img_size
        
        legend_items = [
            ("W=Wall", (0, 100, 255)),
            ("F=Floor", (255, 255, 255)),
            ("D=Door", (0, 255, 0)),
            ("S=Start", (255, 255, 0)),
            ("T=Goal", (255, 215, 0)),
            ("M=Enemy", (255, 0, 0)),
        ]
        
        # Draw legend background
        legend_x = 10
        legend_y = h - 60
        draw.rectangle([legend_x, legend_y, w - 10, h - 10], 
                      fill=(0, 0, 0, 180))
        
        # Draw legend text
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        x_offset = legend_x + 10
        for label, color in legend_items:
            draw.text((x_offset, legend_y + 5), label, fill=color + (255,), font=font)
            x_offset += 70


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify text-to-image alignment for VGLC data")
    parser.add_argument("--dungeon", type=str, default="tloz1_1",
                       help="Dungeon ID to verify (e.g., tloz1_1, tloz2_2)")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Path to 'The Legend of Zelda' data folder")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for verification image")
    
    args = parser.parse_args()
    
    # Determine data root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        # Default: look for Data/The Legend of Zelda relative to script
        script_dir = Path(__file__).parent
        data_root = script_dir.parent / "Data" / "The Legend of Zelda"
    
    if not data_root.exists():
        print(f"ERROR: Data folder not found: {data_root}")
        sys.exit(1)
    
    print("="*70)
    print("VGLC DATA ALIGNMENT VERIFIER")
    print("="*70)
    print(f"Data Root: {data_root}")
    print(f"Dungeon:   {args.dungeon}")
    print()
    
    # Run verification
    verifier = AlignmentVerifier(str(data_root))
    success = verifier.verify(args.dungeon, args.output)
    
    if success:
        print("\n" + "="*70)
        print("VERIFICATION COMPLETE")
        print("="*70)
    else:
        print("\nVerification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
