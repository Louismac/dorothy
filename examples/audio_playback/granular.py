from dorothy import Dorothy
from numpy.random import random

dot = Dorothy()

class MySketch:

    def setup(self): 

        gran_idx = dot.music.start_granular_stream("../audio/gospel.wav", density=12, spread=0.05)
        self.gran = dot.music.audio_outputs[gran_idx]
        #play forever
        self.gran.note_on(440, vel=0.8)

        self.gran.position = 0.7
        self.gran.spread = 0.1
        self.gran.grain_size = 200.0

    def draw(self):
        # Morph parameters live:
        self.gran.spread = 0.03
        self.gran.grain_size = 500.0
        self.gran.n_grains = 1
        self.gran.attack = 0.1
        self.gran.decay = 0.1
        dot.background(dot.darkblue)
        if dot.frames % 1 == 0:
           self.gran.position = random()

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
