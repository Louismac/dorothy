from dorothy.Audio import Sampler
from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        paths = [
            "../audio/snare.wav",
            "../audio/snare2.wav",
            "../audio/meow.wav",
        ]
        self.sampler = Sampler(dot)
        self.sampler.load(paths)
        
        self.clock = dot.music.get_clock()
        self.clock.set_bpm(80)
        
        self.sequence = [1,0,2,0,1,0,0,0,3,0,0,0,1,2,2,2]
        self.clock.on_tick = self.on_tick
        self.clock.play()
        
        self.grid = np.linspace(0,dot.width,len(self.sequence))

    def on_tick(self):
        n = len(self.sequence)
        note = self.sequence[self.clock.tick_ctr % n]
        if note > 0:
            self.sampler.trigger(note-1)
        
    def draw(self):
        dot.background(dot.darkblue)
        y = dot.height/2
        for x in self.grid:
            dot.fill(dot.white)
            dot.circle((x, y), 10)
        #Loop around      
        n = len(self.sequence)
        x = self.grid[self.clock.tick_ctr % n]
        dot.fill(dot.red)
        dot.circle((x, y), 10)
        
MySketch() 