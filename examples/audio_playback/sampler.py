from dorothy import Dorothy
import numpy as np
from dorothy.Audio import Sequence, Note

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
        sampler_idx = dot.music.start_sampler_stream(paths)
        self.sampler = dot.music.audio_outputs[sampler_idx]
        self.clock = dot.music.get_clock(bpm=120)
        self.clock.set_tpb(4) 
        seq = Sequence(steps=16, ticks_per_step=1)
        seq[0] = Note(0, vel=1.0)          
        seq[4] = Note(1, vel=0.8)            
        seq[8] = Note(0, vel=0.9)          
        seq[12] = [Note(1, vel=0.8), Note(2, vel=0.5)]  

        seq.connect(self.clock, self.sampler)
        self.clock.play()
        
    def draw(self):
        dot.background(dot.darkblue)
       
        
MySketch() 