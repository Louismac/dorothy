from dorothy import Dorothy
from random import random
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        dot.fill(dot.blue)
        dot.set_stroke_weight(1)

    def draw(self):
        dot.camera_2d()
        dot.background((50, 50, 50))
        dot.fill((255, 0, 0))
        dot.stroke(dot.green)
        
        # Draw at all four corners
        dot.circle((10, 10), 10)        # Top-left - RED
        dot.circle((630, 10), 10)       # Top-right - should be red
        dot.circle((10, 630), 10)       # Bottom-left - should be red  
        dot.circle((630, 630), 10)      # Bottom-right - should be red
        
        # Center
        dot.fill((0, 255, 0))
        dot.circle((320, 320), 50)  

MySketch()          







