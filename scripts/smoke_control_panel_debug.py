import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from gui_runner import ZeldaGUI

g = ZeldaGUI(maps=[np.zeros((10,10), int)], map_names=['M1'])
# Turn on debug overlay and initialize panel
print('GUI initialized')
g.debug_control_panel = True

# Initialize control panel and update positions
g._init_control_panel()
g._update_control_panel_positions()

print('control_panel_content_height:', g.control_panel_content_height)
print('control_panel_scroll_max:', getattr(g, 'control_panel_scroll_max', None))
print('widget count:', len(g.widget_manager.widgets) if g.widget_manager else 0)

# Toggle debug overlay briefly to print debug info when rendering
# (run GUI main loop for a few frames to see printed debug lines)
for i in range(3):
    g._render()  # call a single frame render (method exists) if available
print('smoke done')
