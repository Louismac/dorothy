from dorothy import Dorothy
from random import random
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        self.bg_layer = dot.get_layer()
        
        # Draw static background once
        with dot.layer(self.bg_layer):
            dot.background(dot.black)
            for i in range(100):
                dot.fill((100,100,100))
                dot.circle((random() * 800, random() * 600), 5)
                dot.fill((0,100,255))
                dot.circle((random() * 800, random() * 600), 5)
                

    def draw(self):       
        # # Draw background layer
        dot.draw_layer(self.bg_layer)
        # Draw foreground
        dot.stroke(dot.green)
        dot.line((0,0),(dot.mouse_x, dot.mouse_y))
        dot.no_stroke()
        pts = np.random.random((1000,2))
        [dot.circle((pt[0] * dot.width, pt[1] * dot.height), 1) for pt in pts]            

MySketch()          







