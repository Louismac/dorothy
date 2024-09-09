from cv2 import circle, rectangle
from src.Dorothy import Dorothy

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
        
        dot.music.play()
        
    def draw(self):        
        for bin_num, bin_val in enumerate(dot.music.fft()[:100:8]):
            circle(dot.canvas, (bin_num*60, int(bin_val*50)), 50, (255,0,255),-1)

        #Cover with a new alpha layer (instead of fully clearing with background)   
        cover=dot.get_layer()
        rectangle(cover, (0,0),(dot.width, dot.height),dot.black, -1)
        dot.draw_layer(cover,0.1)
        

MySketch()          







