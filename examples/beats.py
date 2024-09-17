
from dorothy import Dorothy
dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
        show_beat = 0
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        
    def draw(self):
        col = dot.black
        if dot.music.is_beat():
            self.show_beat = 10
        
        if self.show_beat > 0:
            col = dot.white
        
        dot.background(col)
        self.show_beat -= 1

MySketch()   
    







