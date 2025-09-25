from dorothy import Dorothy 
import numpy as np
from PIL import Image

dot = Dorothy(550, 200) 

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup) 

    def setup(self):
        rgb_image = np.array(Image.open('../images/space.jpg'))
        h = rgb_image.shape[0]
        w = rgb_image.shape[1]
        
        skip = 20
        downsampled = rgb_image[::skip,::skip]
        print(f"skipping leaves us with {downsampled.shape}, we can resize back to original size")
        downsampled = Image.fromarray(downsampled).resize((w,h), Image.NEAREST)
        
        dot.paste(dot.canvas, rgb_image, (0,0))
        dot.paste(dot.canvas, downsampled, (w,0))
          
MySketch()