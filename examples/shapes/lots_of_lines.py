from dorothy import Dorothy
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        dot.stroke(dot.white)
        dot.set_stroke_weight(2)
                

    def draw(self): 
        dot.background(dot.black)
        pts = np.random.random((5000,2))
        with dot.transform():
            # dot.translate(dot.centre[0], dot.centre[1])
            # dot.rotate(dot.frames * 0.02)  # Rotate around Y
            # dot.translate(-dot.centre[0], -dot.centre[1])
            [dot.line((pt[0] * dot.width, pt[1] * dot.height), ((pt[0] * dot.width) + 10, (pt[1] * dot.height)+10)) for pt in pts]            

MySketch()          







