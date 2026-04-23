"""Grid of shapes, each rotating around its own centre."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/disco.wav")
        dot.stroke(dot.grey)

    def draw(self):
        dot.background((22, 208, 165))
        dot.set_stroke_weight(2)
        size, gap, cols = 40, 10, 5
        x_off = (dot.width  - cols * (size + gap)) // 2
        y_off = (dot.height - cols * (size + gap)) // 2
        theta = dot.music.amplitude() * 15 * 2 * np.pi
        for i in range(cols):
            for j in range(cols):
                x  = i * (size + gap) + x_off
                y  = j * (size + gap) + y_off
                cx, cy = x + size / 2, y + size / 2
                with dot.transform():
                    dot.translate(cx, cy)
                    dot.rotate(theta)
                    dot.translate(-cx, -cy)
                    if i % 2 == 0:
                        dot.line((x, y), (cx, cy))
                    else:
                        dot.rectangle((x, y), (x + size, y + size))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
