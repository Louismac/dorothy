from dorothy import Dorothy
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        dot.fill(dot.blue)
        dot.stroke(dot.white)
                

    def draw(self): 
        dot.background(dot.black)
        pts = np.random.random((5000,2))
        [dot.rectangle((pt[0] * dot.width, pt[1] * dot.height), ((pt[0] * dot.width) + 100, (pt[1] * dot.height)+100)) for pt in pts]            

MySketch()          







