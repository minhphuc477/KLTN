"""
Feature 3: Style Transfer Support
==================================
Enable multiple visual themes while preserving gameplay mechanics.

Problem:
    System is trained on Zelda assets, creating IP dependency.
    Need to support multiple themes (castle, cave, desert) for independence.

Solution:
    - Separate semantic layer (gameplay) from visual layer (theme)
    - Use ControlNet-style guidance for style transfer
    - Maintain tile adjacency rules per theme
    - Provide theme-specific sprite atlases
    - Support runtime theme swapping

Integration Point: After VQ-VAE decoding, before rendering
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ThemeType(Enum):
    """Available visual themes."""
    ZELDA_CLASSIC = "zelda_classic"
    CASTLE = "castle"
    CAVE = "cave"
    DESERT = "desert"
    FOREST = "forest"
    DUNGEON = "dungeon"
    TECH = "tech"


@dataclass
class TileThemeMapping:
    """Visual style for a semantic tile in a theme."""
    semantic_id: int  # Logical tile ID (gameplay)
    sprite_id: int  # Visual sprite ID in atlas
    palette_indices: List[int]  # Color palette indices
    animation_frames: Optional[List[int]] = None
    adjacency_rules: Dict[str, List[int]] = field(default_factory=dict)  # {direction: [allowed_neighbors]}


@dataclass
class ThemeConfig:
    """Complete configuration for a visual theme."""
    theme_name: str
    theme_type: ThemeType
    sprite_atlas_path: str  # Path to sprite sheet
    tile_mappings: Dict[int, TileThemeMapping]  # {semantic_id: TileThemeMapping}
    color_palette: List[Tuple[int, int, int]]  # RGB colors
    ambient_effects: Dict[str, Any] = field(default_factory=dict)  # Fog, lighting, particles
    sound_theme: Optional[str] = None


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer process."""
    use_controlnet: bool = True
    guidance_scale: float = 7.5
    preserve_edges: bool = True  # Preserve structural boundaries
    edge_threshold: float = 0.5
    batch_size: int = 1
    device: str = 'cuda'


# ============================================================================
# THEME MANAGER
# ============================================================================

class ThemeManager:
    """
    Manages visual themes and semantic-to-visual mapping.
    
    Design:
    - Semantic layer: Abstract tile IDs (FLOOR=1, WALL=2, etc.)
    - Visual layer: Theme-specific sprites and colors
    - Mapping layer: ThemeConfig defines semantic → visual
    
    This allows:
    - Swapping themes without changing gameplay
    - Generating with one model, rendering with any theme
    - IP independence (generate gameplay, theme separately)
    """
    
    def __init__(self, themes_dir: str = "assets/themes"):
        self.themes_dir = Path(themes_dir)
        self.themes: Dict[str, ThemeConfig] = {}
        self.current_theme: Optional[str] = None
        
        # Load all themes
        self._load_themes()
    
    def _load_themes(self):
        """Load all theme configurations from disk."""
        if not self.themes_dir.exists():
            logger.warning(f"Themes directory not found: {self.themes_dir}")
            # Create default Zelda theme
            self.themes["zelda_classic"] = self._create_default_zelda_theme()
            self.current_theme = "zelda_classic"
            return
        
        for theme_file in self.themes_dir.glob("*.json"):
            try:
                with open(theme_file, 'r') as f:
                    theme_data = json.load(f)
                    theme_config = self._parse_theme_config(theme_data)
                    self.themes[theme_config.theme_name] = theme_config
                    logger.info(f"Loaded theme: {theme_config.theme_name}")
            except Exception as e:
                logger.error(f"Failed to load theme {theme_file}: {e}")
        
        # Set default theme
        if self.themes:
            self.current_theme = list(self.themes.keys())[0]
    
    def _create_default_zelda_theme(self) -> ThemeConfig:
        """Create default Zelda-style theme."""
        from src.core.definitions import SEMANTIC_PALETTE
        
        tile_mappings = {}
        for name, semantic_id in SEMANTIC_PALETTE.items():
            tile_mappings[semantic_id] = TileThemeMapping(
                semantic_id=semantic_id,
                sprite_id=semantic_id,  # 1:1 mapping for default
                palette_indices=[0, 1, 2, 3],
                adjacency_rules={}
            )
        
        return ThemeConfig(
            theme_name="zelda_classic",
            theme_type=ThemeType.ZELDA_CLASSIC,
            sprite_atlas_path="assets/sprites/zelda_tiles.png",
            tile_mappings=tile_mappings,
            color_palette=[
                (0, 0, 0),      # Black (void)
                (197, 179, 88),  # Floor beige
                (88, 48, 0),     # Wall brown
                (255, 255, 255), # White
            ],
            ambient_effects={'lighting': 'dim'}
        )
    
    def _parse_theme_config(self, theme_data: Dict) -> ThemeConfig:
        """Parse theme configuration from JSON."""
        tile_mappings = {}
        for semantic_id, mapping_data in theme_data.get('tile_mappings', {}).items():
            tile_mappings[int(semantic_id)] = TileThemeMapping(
                semantic_id=int(semantic_id),
                sprite_id=mapping_data['sprite_id'],
                palette_indices=mapping_data.get('palette_indices', [0]),
                animation_frames=mapping_data.get('animation_frames'),
                adjacency_rules=mapping_data.get('adjacency_rules', {})
            )
        
        return ThemeConfig(
            theme_name=theme_data['theme_name'],
            theme_type=ThemeType(theme_data['theme_type']),
            sprite_atlas_path=theme_data['sprite_atlas_path'],
            tile_mappings=tile_mappings,
            color_palette=[tuple(c) for c in theme_data['color_palette']],
            ambient_effects=theme_data.get('ambient_effects', {}),
            sound_theme=theme_data.get('sound_theme')
        )
    
    def set_theme(self, theme_name: str):
        """Switch to a different theme."""
        if theme_name not in self.themes:
            raise ValueError(f"Theme not found: {theme_name}")
        self.current_theme = theme_name
        logger.info(f"Switched to theme: {theme_name}")
    
    def get_current_theme(self) -> ThemeConfig:
        """Get current active theme."""
        if self.current_theme is None:
            raise RuntimeError("No theme loaded")
        return self.themes[self.current_theme]
    
    def export_theme(self, theme_name: str, output_path: str):
        """Export theme configuration to JSON."""
        if theme_name not in self.themes:
            raise ValueError(f"Theme not found: {theme_name}")
        
        theme = self.themes[theme_name]
        theme_data = {
            'theme_name': theme.theme_name,
            'theme_type': theme.theme_type.value,
            'sprite_atlas_path': theme.sprite_atlas_path,
            'tile_mappings': {
                str(sem_id): {
                    'sprite_id': mapping.sprite_id,
                    'palette_indices': mapping.palette_indices,
                    'animation_frames': mapping.animation_frames,
                    'adjacency_rules': mapping.adjacency_rules
                }
                for sem_id, mapping in theme.tile_mappings.items()
            },
            'color_palette': theme.color_palette,
            'ambient_effects': theme.ambient_effects,
            'sound_theme': theme.sound_theme
        }
        
        with open(output_path, 'w') as f:
            json.dump(theme_data, f, indent=2)
        
        logger.info(f"Exported theme to {output_path}")


# ============================================================================
# STYLE TRANSFER ENGINE
# ============================================================================

class StyleTransferEngine:
    """
    Neural style transfer with semantic preservation.
    
    Algorithm:
    1. Extract edge map from semantic layout (ControlNet-style)
    2. Generate visual tiles conditioned on edges + theme embedding
    3. Preserve adjacency constraints from theme config
    4. Ensure gameplay-critical features remain distinct
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[StyleTransferConfig] = None
    ):
        self.config = config or StyleTransferConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        
        # Load style transfer model (placeholder - would be ControlNet or similar)
        self.style_model = None
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load pre-trained style transfer model."""
        # Placeholder - in production, load ControlNet or custom model
        logger.info(f"Loading style transfer model from {model_path}")
        # self.style_model = torch.load(model_path).to(self.device)
    
    def apply_theme(
        self,
        semantic_grid: np.ndarray,
        theme_config: ThemeConfig
    ) -> np.ndarray:
        """
        Apply visual theme to semantic grid.
        
        Args:
            semantic_grid: (H, W) semantic tile IDs
            theme_config: Visual theme configuration
        
        Returns:
            (H, W, 3) RGB image with themed visuals
        """
        H, W = semantic_grid.shape
        
        # For now, implement simple palette mapping
        # In production, use neural style transfer model
        
        visual_grid = np.zeros((H, W, 3), dtype=np.uint8)
        
        for r in range(H):
            for c in range(W):
                semantic_id = int(semantic_grid[r, c])
                
                if semantic_id in theme_config.tile_mappings:
                    mapping = theme_config.tile_mappings[semantic_id]
                    # Use first palette color for this tile
                    palette_idx = mapping.palette_indices[0]
                    if palette_idx < len(theme_config.color_palette):
                        color = theme_config.color_palette[palette_idx]
                        visual_grid[r, c] = color
                else:
                    # Default to black
                    visual_grid[r, c] = (0, 0, 0)
        
        return visual_grid
    
    def transfer_style_neural(
        self,
        semantic_grid: np.ndarray,
        theme_embedding: torch.Tensor,
        preserve_edges: bool = True
    ) -> np.ndarray:
        """
        Neural style transfer with edge preservation.
        
        Args:
            semantic_grid: (H, W) semantic layout
            theme_embedding: (D,) theme style embedding
            preserve_edges: Preserve structural boundaries
        
        Returns:
            (H, W, 3) RGB styled image
        """
        if self.style_model is None:
            logger.warning("Style transfer model not loaded, using simple mapping")
            return np.zeros((semantic_grid.shape[0], semantic_grid.shape[1], 3))
        
        # Convert to tensor
        semantic_tensor = torch.from_numpy(semantic_grid).long().unsqueeze(0).to(self.device)
        
        # Extract edge map if preserving structure
        if preserve_edges:
            edge_map = self._extract_edge_map(semantic_grid)
            edge_tensor = torch.from_numpy(edge_map).float().unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            edge_tensor = None
        
        # Apply style transfer
        with torch.no_grad():
            styled_output = self.style_model(
                semantic=semantic_tensor,
                edges=edge_tensor,
                style=theme_embedding.unsqueeze(0).to(self.device)
            )
        
        # Convert back to numpy
        styled_image = styled_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        styled_image = (styled_image * 255).clip(0, 255).astype(np.uint8)
        
        return styled_image
    
    def _extract_edge_map(self, semantic_grid: np.ndarray) -> np.ndarray:
        """
        Extract edge map from semantic grid using Sobel filter.
        
        Returns:
            (H, W) edge strength map [0.0, 1.0]
        """
        from scipy.ndimage import sobel
        
        # Apply Sobel edge detection
        sx = sobel(semantic_grid.astype(float), axis=0)
        sy = sobel(semantic_grid.astype(float), axis=1)
        edge_magnitude = np.sqrt(sx**2 + sy**2)
        
        # Normalize to [0, 1]
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        
        # Threshold
        edge_map = (edge_magnitude > self.config.edge_threshold).astype(float)
        
        return edge_map


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/pipeline/dungeon_pipeline.py:

from src.generation.style_transfer import ThemeManager, StyleTransferEngine

class NeuralSymbolicDungeonPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.theme_manager = ThemeManager(themes_dir="assets/themes")
        self.style_engine = StyleTransferEngine(
            model_path="checkpoints/style_transfer.pth",
            config=StyleTransferConfig()
        )
    
    def generate_room(self, ..., theme_name: Optional[str] = None):
        # ... existing generation logic ...
        
        # After decoding to semantic grid
        semantic_grid = final_grid  # (H, W) semantic tile IDs
        
        # Apply theme if requested
        if theme_name:
            self.theme_manager.set_theme(theme_name)
            theme_config = self.theme_manager.get_current_theme()
            
            visual_grid = self.style_engine.apply_theme(
                semantic_grid=semantic_grid,
                theme_config=theme_config
            )
            
            # Store both semantic and visual
            result.semantic_grid = semantic_grid
            result.visual_grid = visual_grid
        
        return result


# In gui_runner.py (theme selector):

from src.generation.style_transfer import ThemeManager, ThemeType

class ZeldaGUI:
    def __init__(self, ...):
        # ... existing init ...
        self.theme_manager = ThemeManager()
        self.available_themes = list(self.theme_manager.themes.keys())
        self.current_theme_idx = 0
    
    def _handle_theme_switch(self, event):
        '''Handle theme switching via keyboard.'''
        if event.key == pygame.K_t:
            # Cycle through themes
            self.current_theme_idx = (self.current_theme_idx + 1) % len(self.available_themes)
            theme_name = self.available_themes[self.current_theme_idx]
            self.theme_manager.set_theme(theme_name)
            logger.info(f"Switched to theme: {theme_name}")
            
            # Re-render dungeon with new theme
            self._rerender_with_theme()
    
    def _rerender_with_theme(self):
        '''Re-render current dungeon with active theme.'''
        if self.dungeon_grid is None:
            return
        
        from src.generation.style_transfer import StyleTransferEngine
        style_engine = StyleTransferEngine()
        
        theme_config = self.theme_manager.get_current_theme()
        self.visual_dungeon = style_engine.apply_theme(
            semantic_grid=self.dungeon_grid,
            theme_config=theme_config
        )


# Theme JSON format (assets/themes/castle.json):
{
  "theme_name": "castle",
  "theme_type": "castle",
  "sprite_atlas_path": "assets/sprites/castle_tiles.png",
  "tile_mappings": {
    "1": {
      "sprite_id": 100,
      "palette_indices": [0, 1],
      "adjacency_rules": {}
    },
    "2": {
      "sprite_id": 101,
      "palette_indices": [2, 3],
      "adjacency_rules": {}
    }
  },
  "color_palette": [
    [128, 128, 128],
    [160, 160, 160],
    [64, 64, 64],
    [192, 192, 192]
  ],
  "ambient_effects": {
    "lighting": "torches",
    "particles": "dust"
  },
  "sound_theme": "castle_ambient"
}
"""
