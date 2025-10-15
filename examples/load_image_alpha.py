from dorothy import Dorothy 
import numpy as np
from PIL import Image

dot = Dorothy(550, 183) 

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw) 

    def setup(self):
        self.rgb_image = Image.open('../images/space.jpg')
        self.mario = Image.open('../images/mario.png')
        self.grayscale = np.array(self.rgb_image)
        self.rgb_image = np.array(self.rgb_image)
        self.layer = dot.get_layer()


    def draw(self):

        #draw background
        w = self.rgb_image.shape[1]
        dot.paste(self.layer, self.grayscale, (0,0))
        dot.paste(self.layer, self.rgb_image, (w,0))

        #convert background to Pillow Image
        layer = Image.fromarray(layer)
        #Paste sprite with transparency
        layer.paste(self.mario, (dot.mouse_x, dot.mouse_y), self.mario)
        
        #convert back to np.array and draw layer
        layer = np.array(layer)
        dot.draw_layer(layer)
          
MySketch()