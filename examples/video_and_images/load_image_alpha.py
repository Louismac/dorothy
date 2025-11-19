from dorothy import Dorothy 
import numpy as np
from PIL import Image

dot = Dorothy(550, 183) 

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw) 

    def setup(self):
        self.space = np.array(Image.open('../images/space.jpg'))
        self.mario = np.array(Image.open('../images/mario.png'))

    def draw(self):
        dot.paste(self.space, (0,0))
        dot.paste(self.mario, (0,0))
          
MySketch()