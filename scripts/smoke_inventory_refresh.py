import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
from gui_runner import ZeldaGUI

# Prepare GUI and simulate background update

g = ZeldaGUI(maps=[np.zeros((10,10),int)], map_names=['M1'])
# Simulate a background change: collected set updated and counts updated off-main-thread

g.env.state.collected_items = {(1,1)}
# item_type_map needs entry so counts can be computed

g.item_type_map[(1,1)] = 'key'

print('before keys_collected', g.keys_collected)
# Set flag to request update
g.inventory_needs_refresh = True
# Call a render frame which should process deferred refresh
g._render()
print('after keys_collected', g.keys_collected)
