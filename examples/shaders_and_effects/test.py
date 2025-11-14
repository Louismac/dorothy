from dorothy import Dorothy
from random import random
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        dot.fill(dot.blue)

    def draw(self):
        print("start draw loop")
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.no_stroke()
        dot.circle((320, 320), 50)  # Dead center of 640x640
        dot.circle((0, 0), 20)      # Top-left corner
        dot.circle((640, 640), 20)  # Bottom-right corner  
        print(f"viewport {dot.renderer.ctx.viewport}")     

MySketch()          







