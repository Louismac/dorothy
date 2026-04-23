"""Polyphonic synth with a clock-driven step sequencer.

The clock ticks at bpm * tpb (ticks per beat). Sequences fire notes
on each step; multiple synths can run in parallel.
"""
from dorothy import Dorothy
from dorothy.Audio import Sequence, Note
from numpy.random import random

dot = Dorothy()

class MySketch:

    def setup(self):
        self.clock = dot.music.get_clock(bpm=120)
        self.clock.set_tpb(8)           # 8 ticks/beat → 8th-note grid

        synth_idx    = dot.music.start_poly_synth_stream()
        self.synth   = dot.music.audio_outputs[synth_idx]
        synth_idx2   = dot.music.start_poly_synth_stream()
        self.synth2  = dot.music.audio_outputs[synth_idx2]

        self.seq  = Sequence(steps=8, ticks_per_step=4)
        self.seq2 = Sequence(steps=8, ticks_per_step=4)

        # Bass line: constant low note on synth2
        self.seq2.set_pattern([[Note(28)] for _ in range(8)])
        self.seq2.connect(self.clock, self.synth2)
        self.seq.connect(self.clock, self.synth)

        self.clock.play()
        self.clock.on_tick_fns.append(self._on_tick)

    def _on_tick(self):
        # Randomise the melody every 32 ticks
        if self.clock.tick_ctr % 32 == 0:
            chords = [[Note(40 + i + j, vel=0.1, duration=2, waveform="sine")
                       for j in [0, 4, 7]]
                      for i in [0, 7, 2, 9, 4, 11, 6]]
            n = 0
            for i in range(self.seq.steps):
                if random() < 0.3:
                    self.seq[i] = chords[n % len(chords)]
                    n += 1
                else:
                    self.seq[i] = []

    def draw(self):
        dot.background(dot.darkblue)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
