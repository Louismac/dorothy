import numpy as np
from cv2 import rectangle
from src.Dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Pass in the original dataset to use as seeds to generation 
        dot.music.start_magnet_stream(model_path="models/magnet_wiley.pth",
                                      dataset_path="../audio/Wiley.wav")
        dot.music.play()
        
    def draw(self):
        dot.background((1,1,1))
        win_size = 10
        scale = 6
        alpha = 0.3
        #Only draw 20 rectangles
        for i in range(20):
            #Get max fft val in window of frequeny bins
            window = dot.music.fft()[i*win_size:(i+1)*win_size]
            val = int(np.max(window))
            width = val*(i*scale)
            top_left = (dot.width//2-width,dot.height//2-width)
            bottom_right = (dot.width//2+width,dot.height//2+width)
            #draw to an alpha layer
            new_layer = dot.get_layer()
            rectangle(new_layer, top_left, bottom_right, (255*val,164*val,226*val), -1)
            dot.draw_layer(new_layer, alpha)

MySketch()          