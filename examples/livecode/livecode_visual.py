"""Live-coding template for visuals: edit and save to hot-reload the sketch."""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.col = (0, 255, 0)

    def run_once(self):
        # Called once after each file save; use for one-shot state changes
        self.col = (0, 0, 0)

    def draw(self):
        dot.background(self.col)
        dot.fill(dot.blue)
        dot.rectangle((0, dot.frames % 40), (400, 100))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
