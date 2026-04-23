"""Polyline (open path) and polygon (filled closed shape)."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(800, 600)

class MySketch:

    def setup(self):
        num_pts = 60
        self.x      = np.linspace(50, dot.width - 50, num_pts)
        self.base_y = np.sin(np.linspace(0, np.pi * 8, num_pts)) * 120

    def draw(self):
        dot.background(dot.white)

        # Animate the wave using dot.frames
        y   = (self.base_y * np.cos(dot.frames * 0.03) + dot.height // 3).astype(np.int32)
        pts = np.column_stack([self.x, y]).astype(np.int32)

        dot.stroke(dot.green)
        dot.set_stroke_weight(3)
        dot.no_fill()
        dot.polyline(pts, closed=False)

        # Pac-man (concave polygon)
        dot.fill(dot.yellow)
        dot.no_stroke()
        dot.polygon([(100, 420), (150, 370), (200, 420), (200, 500), (150, 455)])

        # Star shape (concave)
        star = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            r = 80 if i % 2 == 0 else 40
            star.append((580 + r * np.cos(angle), 450 + r * np.sin(angle)))
        dot.fill(dot.red)
        dot.stroke(dot.white)
        dot.set_stroke_weight(1)
        dot.polygon(star)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
