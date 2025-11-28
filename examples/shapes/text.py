from dorothy import Dorothy
import numpy as np

dot = Dorothy(640,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        pass
            
    def draw(self):
        dot.background(dot.black)
        dot.fill(dot.red)
        with dot.transform():
            dot.translate(dot.centre[0],dot.centre[1])
            dot.scale((dot.frames*0.01)%5)
            dot.translate(-dot.centre[0],-dot.centre[1])
            for i in range(255):
                dot.fill((i, i,i))
                dot.text("Hello World", -150 + np.random.random()*dot.width, np.random.random()*dot.height, 24)
                dot.text(str(i), -150 + np.random.random()*dot.width, np.random.random()*dot.height, 36)

MySketch()          







