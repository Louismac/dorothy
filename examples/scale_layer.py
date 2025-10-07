from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        dot.stroke(dot.red)
        
    def draw(self):
        dot.background((77, 72, 79))
        factor = dot.music.amplitude() * 15 

        #Initially draw line and circle in middle
        new_layer = dot.get_layer()
        top_left = (dot.width//4, dot.height//4)
        bottom_right = (dot.width//4*3, dot.height//4*3)
        centre = np.array([dot.width//2, dot.height//2])
        dot.circle(centre, 100, new_layer)
        dot.line(top_left, bottom_right, new_layer)

        #scale
        new_layer = dot.scale_layer(new_layer, factor, factor, centre)
        dot.draw_layer(new_layer)
        
MySketch()          







