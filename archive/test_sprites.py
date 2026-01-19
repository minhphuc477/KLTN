"""Test sprite loading."""
import sys
sys.path.insert(0, '.')
from gui_runner import ZeldaGUI
import pygame

# Test sprite loading
class TestGUI(ZeldaGUI):
    def __init__(self):
        pygame.init()
        self.TILE_SIZE = 32
        self.images = {}
        # Test loading assets
        ok = self._load_tileset_sprites()
        if ok:
            print("Tileset loading: OK")
        else:
            print("Tileset loading: FAILED (using fallback colors)")
        
        link = self._load_link_sprite()
        print(f'Link sprite size: {link.get_size()}')
        pygame.quit()

test = TestGUI()
