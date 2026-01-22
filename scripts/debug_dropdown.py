import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gui_runner import ZeldaGUI
from src.gui.widgets import DropdownWidget

def debug():
    gui = ZeldaGUI()
    wm = gui.widget_manager
    floor = next(w for w in wm.widgets if getattr(w, 'control_name', None) == 'floor')
    zoom = next(w for w in wm.widgets if getattr(w, 'control_name', None) == 'zoom')
    print('Before: floor.is_open=%s zoom.is_open=%s' % (floor.is_open, zoom.is_open))
    # open floor
    ret1 = wm.handle_mouse_down((floor.rect.centerx, floor.rect.centery), 1)
    print('After opening floor: floor.is_open=%s zoom.is_open=%s ret=%s' % (floor.is_open, zoom.is_open, ret1))
    # click zoom
    ret2 = wm.handle_mouse_down((zoom.rect.centerx, zoom.rect.centery), 1)
    print('After clicking zoom: floor.is_open=%s zoom.is_open=%s ret=%s' % (floor.is_open, zoom.is_open, ret2))
    print('floor.visible=%s zoom.visible=%s' % (floor.visible, zoom.visible))
    print('floor.rect=%s zoom.rect=%s' % (floor.rect, zoom.rect))
    # direct call
    direct = zoom.handle_mouse_down((zoom.rect.centerx, zoom.rect.centery), 1)
    print('Direct zoom.handle_mouse_down returned %s and zoom.is_open=%s' % (direct, zoom.is_open))

if __name__ == '__main__':
    debug()
