from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        file_path = "../audio/metronome_90.wav"
        dot.music.start_file_stream(file_path)
        #get object for running average
        self.mean_amp = dot.get_window(20)
        
    def draw(self):
        # add new value, get smoothed value
        r = self.mean_amp.add(dot.music.amplitude())
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.circle((dot.width//2,dot.height//2),int(r*dot.height))

MySketch()   
    







