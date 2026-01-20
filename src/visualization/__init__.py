"""
KLTN Visualization Module
=========================

Scientific visualization system for Zelda dungeon validation.

Architecture (Modular Design):
------------------------------
- asset_manager: Robust asset loading with procedural fallbacks
- camera: Viewport system for large maps
- replay_engine: Main replay engine for visualizing solutions
- renderer: Core tile rendering with procedural generation
- effects: Visual feedback effects (pop, flash, ripple)
- hud: Modern HUD with semi-transparent panels

Primary Entry Point:
--------------------
    from src.visualization import DungeonReplayEngine, ReplayConfig
    
    # After solving with A*
    engine = DungeonReplayEngine(dungeon_grid, solution_path)
    engine.run()  # Opens visualization window

Legacy Support:
---------------
    from src.visualization import ZeldaRenderer, EffectManager, ModernHUD
    
    # For custom rendering pipelines
    renderer = ZeldaRenderer(tile_size=32)
    effects = EffectManager()
    hud = ModernHUD()

"""

# ==========================================
# NEW MODULAR COMPONENTS (Recommended)
# ==========================================

# Asset Manager - Robust loading with fallbacks
from src.visualization.asset_manager import (
    AssetManager,
    create_asset_manager,
    SEMANTIC_COLORS,
    SEMANTIC_NAMES,
)

# Camera System - Viewport for large maps
from src.visualization.camera import (
    Camera,
    Viewport,
    create_camera_for_map,
)

# Replay Engine - Main visualization entry point
from src.visualization.replay_engine import (
    DungeonReplayEngine,
    ReplayConfig,
    ReplayState,
    replay_solution,
)

# ==========================================
# LEGACY COMPONENTS (For backward compatibility)
# ==========================================

# Core renderer
from src.visualization.renderer import (
    ThemeConfig,
    Vector2,
    ProceduralTileRenderer,
    SpriteManager,
    AnimationController,
    ZeldaRenderer,
    create_renderer,
)

# Visual effects
from src.visualization.effects import (
    EffectState,
    BaseEffect,
    PopEffect,
    FlashEffect,
    TrailEffect,
    RippleEffect,
    PulseEffect,
    ParticleEffect,
    EffectManager,
    ItemCollectionEffect,
    ItemUsageEffect,
    ItemMarkerEffect,
)

# HUD components
from src.visualization.hud import (
    HUDTheme,
    HUDPanel,
    IconDisplay,
    StatusBar,
    InventoryPanel,
    PathAnalysisPanel,
    ControlsPanel,
    ModernHUD,
)

__all__ = [
    # New Components (Primary)
    'AssetManager',
    'create_asset_manager',
    'SEMANTIC_COLORS',
    'SEMANTIC_NAMES',
    'Camera',
    'Viewport',
    'create_camera_for_map',
    'DungeonReplayEngine',
    'ReplayConfig',
    'ReplayState',
    'replay_solution',
    # Legacy Renderer
    'ThemeConfig',
    'Vector2',
    'ProceduralTileRenderer',
    'SpriteManager',
    'AnimationController',
    'ZeldaRenderer',
    'create_renderer',
    # Effects
    'EffectState',
    'BaseEffect',
    'PopEffect',
    'FlashEffect',
    'TrailEffect',
    'RippleEffect',
    'PulseEffect',
    'ParticleEffect',
    'EffectManager',
    'ItemCollectionEffect',
    'ItemUsageEffect',
    'ItemMarkerEffect',
    # HUD
    'HUDTheme',
    'HUDPanel',
    'IconDisplay',
    'StatusBar',
    'InventoryPanel',
    'PathAnalysisPanel',
    'ControlsPanel',
    'ModernHUD',
]
