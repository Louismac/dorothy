"""Basic Dorothy sketch — use this as a starting template.

dot.start_loop(setup, draw) blocks until the window closes.
For live-coding with hot-reload on save, see livecode_seed.py.
"""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        self.col = dot.white

    def draw(self):
        dot.background(self.col)
        dot.fill(dot.blue)
        dot.rectangle((0, dot.frames % 40), (400, 100))

MySketch()
