from cv2 import rectangle
from src.Dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        
        dot.music.play()

    def draw(self):
        
        dot.background((22, 208, 165))

        size = 40
        border = 10
        grid = 5
        x_offset = (dot.width - ((size+border)*grid)) //2
        y_offset = (dot.height - ((size+border)*grid)) //2
        for i in range(grid):
            for j in range(grid):
                #Where to draw the shape?
                x = i * (size+border) + x_offset
                y = j * (size+border) + y_offset

                #get a new canvas
                new_canvas = dot.push_layer()
                #Draw to it
                top_left = (x,y)
                bottom_right = (x+size, y+size)
                rectangle(new_canvas, top_left, bottom_right, (77, 72, 79), -1)

                theta = dot.music.amplitude * 15 * 2 * np.pi
        
                origin = (x+size/2, y+size/2)
                new_canvas = dot.rotate(new_canvas, theta, origin)
                #push it back onto layer stack
                dot.pop_layer(new_canvas)

MySketch()          







