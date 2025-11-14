
from dorothy import Dorothy
dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Play file from your computer (offline beat tracking)
        file_path = "../audio/disco.wav"
        o = dot.music.start_file_stream(file_path, fft_size=512)

        #Pick or just stream from your computer (online streaming beat tracking)
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #o = dot.music.start_device_stream(1)

        dot.music.audio_outputs[o].onset_detector.threshold = 0.5 
        dot.music.audio_outputs[o].analyse_onsets = True
        dot.music.audio_outputs[o].analyse_beats = True

        self.show_beat = 0
        
    def draw(self):
        
        col = dot.black
        if dot.music.is_beat():
            self.show_beat = 10
        
        if self.show_beat > 0:
            col = dot.white
        
        dot.background(col)
        self.show_beat -= 1

MySketch()   
    







