from dorothy import Dorothy 
import numpy as np


dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.camera_2d()
        self.layer = dot.get_layer()
        with dot.layer(self.layer):
            dot.background(dot.blue)
        dot.set_camera((0, 0, 5), (0, 0, 0))
        self.lights = True
    
    def draw(self):
        dot.camera_3d()
        dot.background((30, 30, 40))

        angle = dot.frames * 0.01
        x = 5 * np.cos(angle)
        z = 5 * np.sin(angle)
        y = 10 * np.sin(angle/2)
        dot.renderer.light_pos = (x, y, z)
        if dot.frames%20==0:
            dot.use_lighting(not dot.renderer.use_lighting)

        dot.camera_3d()
        # 3D box (animated)
        with dot.transform():
            dot.translate(np.sin(np.pi * dot.frames * 0.01), 0, 0)
            dot.fill(dot.blue)
            dot.box(1, 1, 1)
            dot.fill(dot.red)
            dot.translate(0,np.sin(np.pi * dot.frames * 0.01), 0)
            dot.sphere(0.5)
        
Example3D()