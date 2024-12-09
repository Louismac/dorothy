import numpy as np
from cv2 import line
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    
    show_beat = False
    beat_ptr = 0
    pattern_layer = dot.get_layer()

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
        self.base_pattern()
        
    def draw(self):

        extra_scale = 1
        if dot.music.is_beat():
            self.show_beat = True
            self.beat_ptr = 0

        if self.show_beat:
            extra_scale = 1.5
            self.beat_ptr += np.pi/15
            if self.beat_ptr > np.pi:
                self.show_beat = False

        new_canvas = dot.get_layer()
        new_canvas = self.pattern_layer.copy()
        factor = (np.sin(self.beat_ptr)+1)*extra_scale
        origin = (dot.width//2, dot.height//2)
        new_canvas = dot.scale_layer(new_canvas, factor, factor, origin)
        dot.draw_layer(new_canvas)
    
    #Draw the vera molnar grid to the pattern_layer (this gets transformed later)
    def base_pattern(self):
        dot.stroke((255, 37, 21))
        size = 30
        dot.background((208, 184, 158))
        for i in range(dot.width//size):
            for j in range(dot.height//size):
                y1 = j*size
                if np.random.random()<0.5:
                    y1 = (j+1)*size
                y2 = j*size
                if np.random.random()<0.5:
                    y2 = (j+1)*size

                dot.line((i*size,y1), ((i+1)*size,y2), layer = self.pattern_layer) 
        
MySketch()         







