"""Rotate a rectangle around its centre, driven by audio amplitude."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav", buffer_size=2048)
        dot.stroke(dot.grey)

    def draw(self):
        dot.background((22, 208, 165))
        cx, cy = dot.width // 2, dot.height // 2
        # Amplitude (0–1) mapped to full rotation
        theta = dot.music.amplitude() * 3 * 2 * np.pi
        with dot.transform():
            dot.translate(cx, cy)
            dot.rotate(theta)
            dot.translate(-cx, -cy)
            dot.rectangle((dot.width // 4, dot.height // 4),
                           (dot.width * 3 // 4, dot.height * 3 // 4))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
