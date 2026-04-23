"""Scale a pre-drawn layer by audio amplitude each frame.

The layer is regenerated every 100 frames to show a new random pattern.
"""
from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav", fft_size=512)
        self.pattern_layer = dot.get_layer()
        self._draw_pattern()

    def _draw_pattern(self):
        dot.background(dot.beige)
        with dot.layer(self.pattern_layer):
            dot.background((0, 0, 0, 0))
            dot.stroke((255, 37, 21))
            dot.set_stroke_weight(1)
            size = 30
            for i in range(dot.width // size):
                for j in range(dot.height // size):
                    y1 = (j + (1 if np.random.random() < 0.5 else 0)) * size
                    y2 = (j + (1 if np.random.random() < 0.5 else 0)) * size
                    dot.line((i * size, y1), ((i + 1) * size, y2))

    def draw(self):
        if dot.frames % 100 == 0:
            self._draw_pattern()

        factor = dot.music.amplitude() * 15
        cx, cy = dot.width // 2, dot.height // 2
        with dot.transform():
            dot.translate(cx, cy)
            dot.scale(factor)
            dot.translate(-cx, -cy)
            dot.draw_layer(self.pattern_layer)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
