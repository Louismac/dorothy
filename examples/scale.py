from cv2 import rectangle
from Dorothy import Dorothy

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
        
        dot.background((77, 72, 79))
        #get a new canvas
        new_canvas = dot.push_layer()
        #Draw to it
        top_left = (dot.width//4, dot.height//4)
        bottom_right = (dot.width//4*3, dot.height//4*3)
        rectangle(new_canvas, top_left, bottom_right,(22, 208, 165), -1)

        factor = dot.music.amplitude * 15 
   
        origin = (dot.width//2,dot.height//2)
        new_canvas = dot.scale(new_canvas, factor, factor, origin)
        #push it back onto layer stack
        dot.pop_layer(new_canvas)
        

MySketch()          







