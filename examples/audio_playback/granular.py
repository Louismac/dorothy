from dorothy.Audio import Sequence, Note, HighPassFilter, Reverb
from dorothy import Dorothy
from numpy.random import random

dot = Dorothy()

class MySketch:

    def setup(self):
        self.clock = dot.music.get_clock(bpm=120)
        self.clock.set_tpb(4)           # 4 ticks/beat → 16th-note grid

        gran_idx = dot.music.start_granular_stream("../audio/gospel.wav", density=12, spread=0.05)
        self.gran = dot.music.audio_outputs[gran_idx]

        seq = Sequence(steps=1, ticks_per_step=1)
        seq[0] = Note(69, vel=0.7, duration=1)   # A4 = original pitch
        seq.connect(self.clock, self.gran)
        self.clock.play()

        # Morph parameters live:
        self.gran.position = 0.7
        self.gran.spread = 0.1
        self.gran.grain_size = 200.0
        # self.gran.add_effect(Reverb(wet=0.25))

    def draw(self):
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
