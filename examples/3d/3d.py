from dorothy import Dorothy 
import numpy as np

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("3D Setup!")
        
        dot.set_camera((0, 0, 5), (0, 0, 0))
    
    def draw(self):
        dot.background((30, 30, 40))
        
        # 3D box (animated)
        with dot.transform():
            dot.translate(np.sin(np.pi * dot.frames * 0.01), 0, 0)
            dot.fill((100, 100, 255, 128))
            dot.box(1, 1, 1)

        # Switch to 2D for rectangle
        dot.camera_2d()

        # Animate rectangle in 2D (screen coordinates)
        with dot.transform():
            x_offset = 100 * np.sin(np.pi * dot.frames * 0.01)  # Scale to pixels
            dot.translate(x_offset, 0, 0)
            dot.rectangle((0, 0), (300, 300))

        dot.camera_3d()
        # 3D sphere (static)
        dot.fill((255, 100, 100, 128))
        dot.sphere(1.0, (0, 0, 0))
        
Example3D()