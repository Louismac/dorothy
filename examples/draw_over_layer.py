from dorothy import Dorothy
from random import random

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
                dot.fill((100, 100, 200))
                dot.circle((random() * 800, random() * 600), 5)

    def draw(self):        
        # Draw background layer
        dot.draw_layer(self.bg_layer)
        
        # Draw foreground
        dot.fill((255, 255, 0))
        dot.circle((dot.mouse_x, dot.mouse_y), 50)

MySketch()          







