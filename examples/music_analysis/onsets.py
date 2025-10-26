from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        file_path = "../audio/disco.wav"
        o = dot.music.start_file_stream(file_path, fft_size=512)
        #adjust threshold if you want (0.3 default)
        # dot.music.audio_outputs[o].onset_detector.threshold = 0.3
        # self.prev_millis = dot.millis
        # dot.music.start_device_stream(1)
        
    def draw(self):
        dot.background(dot.white)
        if dot.music.is_onset():
            print("Recent onset!")
            dot.fill(dot.red)
            dot.circle((dot.width//2,dot.height//2),int(dot.music.amplitude()*dot.height*10))
        self.prev_millis = dot.millis

MySketch()   
    







