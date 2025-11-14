from dorothy import Dorothy
from random import random
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        dot.fill(dot.blue)
        dot.no_stroke()
                

    def draw(self): 
        dot.camera_2d()
        print(f"Transform matrix: {dot.renderer.transform.matrix}")
        pts = np.random.random((1000,2))
        [dot.circle((pt[0] * dot.width, pt[1] * dot.height), 2) for pt in pts]            

MySketch()          







