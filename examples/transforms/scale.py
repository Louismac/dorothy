"""Scale shapes from their centre, driven by audio amplitude."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav")
        dot.stroke(dot.red)

    def draw(self):
        dot.background((77, 72, 79))
        factor = dot.music.amplitude() * 15
        cx, cy = dot.width // 2, dot.height // 2
        with dot.transform():
            # Translate to centre, scale, then translate back
            dot.translate(cx, cy)
            dot.scale(factor)
            dot.translate(-cx, -cy)
            dot.line((dot.width // 4, dot.height // 4),
                     (dot.width * 3 // 4, dot.height * 3 // 4))
            dot.circle((cx, cy), 100)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
