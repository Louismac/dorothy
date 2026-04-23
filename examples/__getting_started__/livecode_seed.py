"""Live-coding template — the sketch hot-reloads every time you save.

run_once() is called exactly once after each reload, useful for
one-time state changes without re-running the full setup.
"""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        self.col = (0, 255, 0)

    def run_once(self):
        # Executes once after each file save; use for one-shot changes
        self.col = (255, 5, 200)

    def draw(self):
        dot.background(self.col)
        dot.fill(dot.red)
        dot.rectangle((0, dot.frames % 40), (400, 100))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
