from dorothy import Dorothy
from dorothy.Audio import Sampler, Clock
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
        
        self.clock = Clock()
        self.clock.set_bpm(80)
        
        self.sequence = [1,1,1,1,1,0,0,0,2,0,0,0,2,0,0,0]
        self.clock.on_tick = self.on_tick
        self.clock.play()

    def on_tick(self):
        n = len(self.sequence)
        note = self.sequence[self.clock.tick_ctr % n]
        if note > 0:
            self.sampler.trigger(note-1)
        
    def draw(self):
        dot.background(dot.white)

        n = len(self.sequence)
        seq_ptr = self.clock.tick_ctr % n
        x = seq_ptr * dot.width/n

        dot.stroke(dot.red)
        dot.line((x, 0),(x,200))
        
MySketch()  

