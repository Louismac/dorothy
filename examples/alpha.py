import numpy as np
from cv2 import rectangle, circle
from Dorothy import Dorothy
import sounddevice as sd

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Play file from your computer
        file_path = "../audio/hiphop.wav"
        dot.music.load_file(file_path, fft_size=512)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        # dot.music.get_stream(2)
        
        dot.music.play()
        
    def draw(self):
        dot.background((226, 226, 43))
        win_size = 10
        scale = 100
        alpha = 0.4
        #Only draw 20 rectangles
        for i in range(20):
            #Get max fft val in window of frequeny bins
            window = dot.music.fft_vals[i*win_size:(i+1)*win_size]
            bottom_corner = int(np.max(window)*(i*scale))
            #draw to an alpha layer
            new_layer = dot.to_alpha(alpha)
            rectangle(new_layer, (0,0), (bottom_corner,bottom_corner), (1,1,1), -1)
        #Call this when you want to render the alpha layers to the canvas (e.g. to draw something else on top of them)
        dot.update_canvas()
        top_left = (dot.width//2-100,dot.height//2-100)
        bottom_right = (dot.width//2+100,dot.height//2+100)
        rectangle(dot.canvas, top_left, bottom_right, (0,0,0), -1)

MySketch()          







