"""Alpha blending: the 4th value in (R, G, B, A) controls opacity (0–255)."""
from dorothy import Dorothy

dot = Dorothy(640, 640)

class MySketch:

    def setup(self):
        pass

    def draw(self):
        dot.background(dot.black)
        # Nested rectangles, each slightly smaller and more transparent
        for i in range(10):
            t = 1 - i / 10
            w = t * dot.width
            x0 = (dot.width  - w) // 2
            y0 = (dot.height - w) // 2
            dot.fill((int(255 * t), int(164 * t), int(226 * t), 120))
            dot.rectangle((x0, y0), (x0 + w, y0 + w))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
