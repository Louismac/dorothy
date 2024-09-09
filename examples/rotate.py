from cv2 import line, circle
from src.Dorothy import Dorothy
from numpy import pi

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
        
        dot.music.play()

    def draw(self):
        
        dot.background((22, 208, 165))
        #get a new canvas
        circle(dot.canvas, (dot.width//2, dot.height//2), dot.height//4, dot.white, 3)
        new_canvas = dot.get_layer()
        #Draw to it
        top_left = (dot.width//4, dot.height//4)
        bottom_right = (dot.width//2, dot.height//2)
        line(new_canvas, top_left, bottom_right, (77, 72, 79), 2)

        theta = dot.music.amplitude() * 3 * 2 * pi
   
        origin = (dot.width//2,dot.height//2)
        new_canvas = dot.rotate(new_canvas, theta, origin)
        #push it back onto layer stack
        dot.draw_layer(new_canvas)
        

MySketch()          







