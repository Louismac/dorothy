import numpy as np
from cv2 import rectangle, circle
from src.Dorothy import Dorothy
import sounddevice as sd

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Play file from your computer
        # file_path = "../audio/disco.wav"
        # dot.music.start_file_stream(file_path, fft_size=512)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        dot.music.start_device_stream(3)
        
        dot.music.play()
        
    def draw(self):
        dot.background((1,1,1))
        win_size = 10
        scale = 6
        alpha = 0.3
        #Only draw 20 rectangles
        for i in range(20):
            #Get max fft val in window of frequeny bins
            window = dot.music.fft_vals[i*win_size:(i+1)*win_size]
            val = int(np.max(window))
            width = val*(i*scale)
            top_left = (dot.width//2-width,dot.height//2-width)
            bottom_right = (dot.width//2+width,dot.height//2+width)
            #draw to an alpha layer
            new_layer = dot.to_alpha(alpha)
            rectangle(new_layer, top_left, bottom_right, (255*val,164*val,226*val), -1)
        #Call this when you want to render the alpha layers to the canvas (e.g. to draw something else on top of them)
        dot.update_canvas()
        top_left = (dot.width//2-10,dot.height//2-10)
        bottom_right = (dot.width//2+10,dot.height//2+10)
        rectangle(dot.canvas, top_left, bottom_right, (255,255,255), -1)

MySketch()          







