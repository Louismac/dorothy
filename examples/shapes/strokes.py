"""Stroke colour and weight — outlines with no fill."""
from dorothy import Dorothy

dot = Dorothy(800, 600)

class MySketch:

    def setup(self):
        pass

    def draw(self):
        dot.background(dot.black)
        dot.no_fill()
        for i, weight in enumerate([1, 2, 4, 8, 16]):
            y = 80 + i * 100
            dot.stroke((255, 255 - i * 40, i * 40))
            dot.set_stroke_weight(weight)
            dot.line((50, y), (dot.width - 50, y))
            dot.circle((dot.width // 2, y), 35)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
