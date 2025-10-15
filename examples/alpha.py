import numpy as np
from dorothy import Dorothy
import sounddevice as sd

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        pass
            
    def draw(self):
        dot.background(dot.black)
        for i in range(10):
            val = i/10
            width = val*dot.width
            top_left = ((dot.width-width)//2,(dot.height-width)//2)
            bottom_right = ((dot.width+width)//2,(dot.height+width)//2)
            print(val, i, width, top_left, bottom_right)
            dot.fill((255*val,164*val,226*val,128))
            dot.rectangle(top_left, bottom_right)

MySketch()          







