import numpy as np
from cv2 import circle
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        self.color = dot.red
        def mouse_pressed(x,y,b):
            if self.color == dot.red:
                self.color = dot.blue
            else:
                self.color = dot.red
            print("HERERERE",x,y,b)
        dot.on_mouse_press = mouse_pressed

    
    def draw(self):
        dot.fill(self.color)
        dot.circle((dot.mouse_x, dot.mouse_y), 100)

MySketch()          