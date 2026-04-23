"""Sample player driven by a step sequencer.

Note index = which sample path to play (0-based).
Velocity controls loudness.
"""
from dorothy import Dorothy
from dorothy.Audio import Sequence, Note

dot = Dorothy()

class MySketch:

    def setup(self):
        paths = [
            "../audio/snare.wav",
            "../audio/snare2.wav",
            "../audio/meow.wav",
        ]
        sampler_idx   = dot.music.start_sampler_stream(paths)
        self.sampler  = dot.music.audio_outputs[sampler_idx]
        self.clock    = dot.music.get_clock(bpm=120)
        self.clock.set_tpb(4)            # 4 ticks per beat = 16th-note grid

        seq = Sequence(steps=16, ticks_per_step=1)
        seq[0]  = Note(0, vel=1.0)       # sample 0, beat 1
        seq[4]  = Note(1, vel=0.8)       # sample 1, beat 2
        seq[8]  = Note(0, vel=0.9)       # sample 0, beat 3
        seq[12] = [Note(1, vel=0.8), Note(2, vel=0.5)]   # two hits on beat 4

        seq.connect(self.clock, self.sampler)
        self.clock.play()

    def draw(self):
        dot.background(dot.darkblue)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
