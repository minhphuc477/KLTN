"""
KLTN GUI Module
===============

Interactive GUI components for the Zelda dungeon visualization system.

Components:
- widgets: Core interactive widgets (CheckboxWidget, DropdownWidget, ButtonWidget, WidgetManager)
- tier2_components: Advanced GUI components (FloorSelector, MinimapZoom, etc.)
- components/: Canonical shared constants/fallbacks/widget modules
- control_panel/: Canonical control-panel layout/render/interaction helpers
- map/: Canonical map-loading/navigation/minimap/viewport helpers
- solver/: Canonical solver flow/orchestration/recovery helpers
- rendering/: Canonical overlays/sidebar/status/render helper modules
- runtime/: Canonical runtime/display/route/temp/toast helpers
- gameplay/: Canonical path/inventory/auto-manual gameplay helpers
- ai/: Canonical AI generation control/pipeline/worker modules
- topology/: Canonical topology export/helper/precheck/matching modules

Legacy top-level modules are kept as backward-compatible shims during incremental refactors.
"""

from .components.widgets import (
    WidgetState,
    WidgetTheme,
    BaseWidget,
    CheckboxWidget,
    DropdownWidget,
    ButtonWidget,
    WidgetManager,
)

__all__ = [
    'WidgetState',
    'WidgetTheme',
    'BaseWidget',
    'CheckboxWidget',
    'DropdownWidget',
    'ButtonWidget',
    'WidgetManager',
]
