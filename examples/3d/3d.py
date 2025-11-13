from dorothy import Dorothy 
import numpy as np
from PIL import Image

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.camera_3d()
        dot.set_camera((40, 40, 40), (0, 0, 0))

    def draw(self):
        dot.background((10, 10, 20))
        dot.camera_3d()
        dot.fill(dot.magenta)
        with dot.transform():
            dot.rotate(dot.frames * 0.01, 0, 1, 0)
            dot.rotate(dot.frames * 0.005, 1, 0, 0)
            dot.box((9, 15, 10))
        dot.fill(dot.cyan)
        with dot.transform():
            dot.rotate(dot.frames * 0.01, 0, 1, 0)
            dot.rotate(dot.frames * 0.005, 1, 0, 0)
            dot.box((15, 10, 11))
        
        
        
Example3D()