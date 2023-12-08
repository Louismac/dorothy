from cv2 import line
from Dorothy import Dorothy
import sounddevice as sd

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
            
    def setup(self):

        #Play file from your computer
        file_path = "../audio/hiphop.wav"
        dot.music.load_file(file_path)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        # dot.music.get_stream(2)
        
        dot.music.play()
        
    def draw(self):
        dot.background((0))
        for i,val in enumerate(dot.music.fft_vals[::8]):
            line(dot.canvas, (i*5, dot.height), (0, dot.height-int(val*450)), (0,(1-val)*255,0), 1+int(10*val))

MySketch()






