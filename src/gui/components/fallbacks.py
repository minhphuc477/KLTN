"""No-op GUI fallbacks used when optional dependencies are unavailable."""

from __future__ import annotations

from typing import Any, Dict


def _noop_effect(*_args: Any, **_kwargs: Any) -> None:
    """Return None for optional visual effects in no-op mode."""
    return None


class _NoOpSpriteManager:
    def __init__(self, pygame_available: bool, pygame_module: Any):
        self._pygame_available = bool(pygame_available)
        self._pygame = pygame_module

    def get_tile(self, _tile_id: int, size: int):
        if self._pygame_available and self._pygame is not None:
            surf = self._pygame.Surface((size, size))
            surf.fill((80, 80, 80))
            return surf
        return None


class NoOpRenderer:
    def __init__(self, tile_size: int, *, pygame_available: bool = False, pygame_module: Any = None):
        self.tile_size = tile_size
        self.sprite_manager = _NoOpSpriteManager(pygame_available, pygame_module)
        self.agent_visual_pos = None
        self.show_heatmap = False

    def set_agent_position(self, *_, **__):
        return None

    def set_tile_size(self, *_, **__):
        return None

    def update(self, *_, **__):
        return None

    def render(self, *_, **__):
        return None


class NoOpEffectManager:
    def add_effect(self, *_, **__):
        return None

    def clear(self):
        return None

    def update(self, *_, **__):
        return None

    def render(self, *_, **__):
        return None


class NoOpHUDInventory:
    def __init__(self):
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0
        self.keys_used = 0
        self.bombs_used = 0
        self.boss_keys_used = 0


class NoOpModernHUD:
    def __init__(self):
        self.inventory = NoOpHUDInventory()
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0
        self.keys_used = 0
        self.bombs_used = 0
        self.boss_keys_used = 0

    def update_game_state(self, *_, **__):
        return None


class NoOpPathPreviewDialog:
    def __init__(self, *_, **__):
        pass

    def render(self, *_, **__):
        return None

    def render_path_overlay(self, *_, **__):
        return None

    def handle_input(self, *_, **__):
        return None


class NoOpWidget:
    def __init__(self, *_, **__):
        self.control_name = None
        self.is_open = False
        self.state = None
        self.rect = None
        self.checked = False

    def handle_mouse_down(self, *_, **__):
        return False

    def handle_mouse_up(self, *_, **__):
        return False


class NoOpWidgetManager:
    def __init__(self):
        self.widgets = []

    def add_widget(self, widget: Any):
        self.widgets.append(widget)

    def snapshot_dropdown_state(self):
        return {}

    def apply_dropdown_state(self, _state: Dict[str, Any]):
        return None

    def update(self, *_, **__):
        return None

    def handle_mouse_down(self, *_, **__):
        return False

    def handle_mouse_up(self, *_, **__):
        return False

    def handle_mouse_motion(self, *_, **__):
        return False


def get_visualization_fallbacks(*, pygame_available: bool, pygame_module: Any) -> Dict[str, Any]:
    """Return symbol bindings expected by gui_runner when visualization imports fail."""

    class _NoOpRendererWithPygame(NoOpRenderer):
        def __init__(self, tile_size: int):
            super().__init__(
                tile_size,
                pygame_available=pygame_available,
                pygame_module=pygame_module,
            )

    return {
        "ZeldaRenderer": _NoOpRendererWithPygame,
        "ThemeConfig": None,
        "Vector2": None,
        "EffectManager": NoOpEffectManager,
        "PopEffect": _noop_effect,
        "FlashEffect": _noop_effect,
        "RippleEffect": _noop_effect,
        "ItemCollectionEffect": _noop_effect,
        "ItemUsageEffect": _noop_effect,
        "ItemMarkerEffect": _noop_effect,
        "ModernHUD": NoOpModernHUD,
        "HUDTheme": None,
        "PathPreviewDialog": NoOpPathPreviewDialog,
    }


def get_widget_fallbacks() -> Dict[str, Any]:
    """Return symbol bindings expected by gui_runner when widget imports fail."""
    return {
        "CheckboxWidget": NoOpWidget,
        "DropdownWidget": NoOpWidget,
        "ButtonWidget": NoOpWidget,
        "WidgetManager": NoOpWidgetManager,
        "WidgetTheme": None,
    }
