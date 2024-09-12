import numpy as np
from cv2 import circle
from dorothy import Dorothy

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
        dot.background((0))
        
        color = (255,255,255)
        if dot.mouse_down:
            color = (255,0,255)
        
        size = dot.music.amplitude()*1000

        circle(dot.canvas,(int(dot.mouse_x), int(dot.mouse_y)),int(size) ,color,-1)

MySketch()          