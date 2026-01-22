from gui_runner import ZeldaGUI
import time
import pygame

g = ZeldaGUI()
# Start collapsed and then expand
g.control_panel_width_current = 40
g._start_toggle_panel_animation(False)
frames = []
for i in range(20):
    g._update_control_panel_animation()
    g._render()
    # save screenshot
    pygame.image.save(g.screen, f"capture_frame_{i:02d}.png")
    time.sleep(0.02)
print('done')
