from cv2 import line
from src.Dorothy import Dorothy
import sounddevice as sd

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
            
    def setup(self):

        #Play file from your computer
        #file_path = "../audio/disco.wav"
        #dot.music.start_file_stream(file_path, fft_size=512)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        dot.music.start_device_stream(4)
        
        dot.music.play()
        
    def draw(self):
        
        dot.background((0))

        for bin_num, bin_val in enumerate(dot.music.fft_vals[::8]):
           
            pt1 = (bin_num*5, dot.height)
            pt2 = (0, dot.height-int(bin_val*1000))
            color = (0,(1-bin_val)*255,0)
            thickness = 1+int(10*bin_val)
           
            line(dot.canvas, pt1, pt2, color, thickness)

MySketch()






