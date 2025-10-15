from dorothy import Dorothy 
import numpy as np
import math

dot = Dorothy(1080,960)

class Example3D:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("3D Setup!")
        dot.camera_3d()
        self.angle = 0
        dot.set_camera((0, 0, 5), (0, 0, 0))
    
    def draw(self):
        dot.background((30, 30, 40))

        angle = dot.frames * 0.01
        x = 5 * math.cos(angle)
        z = 5 * math.sin(angle)
        dot.set_camera((x, 2, z), (0, 0, 0))
        
        # 3D box (animated)
        dot.push_matrix()
        dot.translate(np.sin(np.pi * dot.frames * 0.01)*5, 0, 0)
        dot.fill((100, 100, 255, 128))
        dot.box((1, 1, 1), (0.5, 0, 0))
        dot.pop_matrix()

        # 3D sphere (static)
        dot.fill((255, 100, 100, 128))
        dot.sphere(1.0, (0, 0, 0))
        
Example3D()