from sampler import Sampler
from dorothy import Dorothy
dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        paths = [
            "../audio/snare.wav",
            "../audio/snare2.wav"
        ]
        self.sampler = Sampler(dot)
        self.sampler.load(paths)
        self.sampler.sequence = [
            [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0],
            [1,0,0,1,0,0,1,0]
            ]
        self.sampler.set_bpm(80)
        
    def draw(self):
        is_tick = self.sampler.tick(dot.millis)
        if is_tick:
            if self.sampler.tick_ctr % 32 == 0:
                self.sampler.set_bpm(80)
            if self.sampler.tick_ctr % 32 == 16:
                self.sampler.set_bpm(160)
        
MySketch()  

