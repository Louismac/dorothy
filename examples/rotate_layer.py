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
        dot.music.start_file_stream(file_path, buffer_size=2048)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        dot.stroke(dot.grey)
        
    def draw(self):
        
        new_layer = dot.get_layer()
        dot.background((22, 208, 165))
        theta = dot.music.amplitude() * 3 * 2 * np.pi
        centre = np.array([dot.width//2, dot.height//2])
        
        #Initially draw line in middle
        top_left = (dot.width//4, dot.height//4)
        bottom_right = (dot.width//4*3, dot.height//4*3)
        centre = np.array([dot.width//2, dot.height//2])
        dot.rectangle(top_left, bottom_right, layer = new_layer)
        new_layer = dot.rotate_layer(new_layer, theta, centre)
        dot.draw_layer(new_layer)

MySketch()          







