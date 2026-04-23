"""LFOs (low-frequency oscillators) for animating visual parameters.

get_lfo() returns an ID; lfo_value() reads the current value each frame.
Available waveforms: 'sine', 'saw', 'triangle', 'square'.
"""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        self.size_lfo  = dot.get_lfo('sine',     freq=0.5, range=(20, 100))
        self.x_lfo     = dot.get_lfo('saw',      freq=1.0, range=(0, dot.width))
        self.color_lfo = dot.get_lfo('triangle', freq=2.0, range=(0, 255))
        self.rot_lfo   = dot.get_lfo('sine',     freq=0.3, range=(0, 2 * np.pi))

    def draw(self):
        dot.background(dot.black)
        size  = dot.lfo_value(self.size_lfo)
        x     = dot.lfo_value(self.x_lfo)
        color = int(dot.lfo_value(self.color_lfo))
        rot   = dot.lfo_value(self.rot_lfo)
        dot.fill((color, 100, 255))
        with dot.transform():
            dot.translate(x, dot.height // 2)
            dot.rotate(rot)
            dot.circle((0, 0), size)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
