import numpy as np
from dorothy import Dorothy
import sounddevice as sd

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Play file from your computer
        file_path = "../audio/drums.wav"
        dot.music.start_file_stream(file_path)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        # dot.music.start_device_stream(1)
                
    def draw(self):
        dot.background(dot.black)
        win_size = 10
        scale = 5
        alpha = 0.3
        #Only draw 20 rectangles
        for i in range(20):
            #Get max fft val in window of frequeny bins
            window = dot.music.fft()[i*win_size:(i+1)*win_size]
            val = int(np.max(window))
            width = val*((i+1)*scale)
            top_left = (dot.width//2-width,dot.height//2-width)
            bottom_right = (dot.width//2+width,dot.height//2+width)
            # print(val, i, scale, width, top_left, bottom_right)
            new_layer = dot.get_layer()
            dot.fill((255*val,164*val,226*val))
            dot.rectangle(top_left, bottom_right,  layer = new_layer)
            dot.draw_layer(new_layer, alpha)

MySketch()          







