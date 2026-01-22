from gui_runner import ZeldaGUI
import itertools

g=ZeldaGUI()
print('widgets', len(g.widget_manager.widgets))
# Simulate expanded panel
g.control_panel_width_current=g.control_panel_width
g._update_control_panel_positions()
print('panel rect', g.control_panel_rect)
for i,w in enumerate(itertools.islice(g.widget_manager.widgets,6)):
    print(i, type(w).__name__, 'pos', getattr(w,'pos',None), 'rect', w.rect, 'inside', g.control_panel_rect.colliderect(w.rect))
