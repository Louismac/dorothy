"""Batch-draw thousands of shapes with a list comprehension."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(640, 640)

class MySketch:

    def setup(self):
        dot.no_stroke()

    def draw(self):
        dot.background(dot.black)
        dot.fill(dot.blue)
        pts = np.random.random((2000, 2))
        # List comprehension batches all calls into a single GPU flush
        [dot.circle((pt[0] * dot.width, pt[1] * dot.height), 3) for pt in pts]

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
