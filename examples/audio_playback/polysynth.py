from dorothy.Audio import Sequence, Note
from dorothy import Dorothy
from numpy.random import random

dot = Dorothy()

class MySketch:

    def setup(self):
        self.clock = dot.music.get_clock(bpm=120)
        self.clock.set_tpb(4)           # 4 ticks/beat → 16th-note grid

        synth_idx = dot.music.start_poly_synth_stream()
        self.synth = dot.music.audio_outputs[synth_idx]

        # 8 quarter-note steps
        self.seq = Sequence(steps=8, ticks_per_step=4)
        

        self.seq.connect(self.clock, self.synth)
        self.clock.play()

    def run_once(self):
        print("here")

    def draw(self):
        self.clock.set_tpb(12) 
        self.seq.steps = 16
        dot.background(dot.darkblue)
        if dot.frames % 300 == 0:
            notes = [[Note(40 + i + j, vel=0.1, duration=2, waveform="pwm") for j in [0,4,7]] for i in [0,2,4,5,7,9,11]]
            n = 0
            for i in range(self.seq.steps):
                if random()< 0.5:
                    self.seq[i] = notes[n%len(notes)]
                    n = n + 1
                else:
                    self.seq[i] = []


if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
