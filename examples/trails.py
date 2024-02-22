from cv2 import circle
from src.Dorothy import Dorothy

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
        #second argument to dot.background is alpha value
        dot.background((0,0,0),0.1)
        for bin_num, bin_val in enumerate(dot.music.fft_vals[:100:8]):
            circle(dot.canvas, (bin_num*60, int(bin_val*50)), 50, (255,0,255),-1)
        

MySketch()          







