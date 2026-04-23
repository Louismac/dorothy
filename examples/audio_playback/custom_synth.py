"""DSP stream: provide a raw audio callback for fully custom synthesis.

Mouse X controls frequency (Hz), mouse Y controls amplitude.
"""
import numpy as np
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.phase = 0.0
        sr = 44100

        def get_frame(size):
            freq  = dot.mouse_x
            amp   = dot.mouse_y / dot.height
            inc   = 2 * np.pi * freq / sr
            audio = amp * np.sin(self.phase + inc * np.arange(size))
            self.phase += inc * size
            return audio

        dot.music.start_dsp_stream(get_frame, sr=sr, buffer_size=512)

    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.circle((dot.width // 2, dot.height // 2),
                   int(dot.music.amplitude() * dot.height * 10))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
