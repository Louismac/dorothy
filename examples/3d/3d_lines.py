"""3D line and polyline drawing: a wireframe cube and a helix."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.camera_3d()
        dot.set_camera((5, 5, 5), (0, 0, 0))
        self.helix = [(np.cos(i * 0.2), i / 100, np.sin(i * 0.2))
                      for i in range(100)]

    def _draw_cube(self, size):
        s = size / 2
        v = [(-s,-s,-s),(s,-s,-s),(s,s,-s),(-s,s,-s),
             (-s,-s, s),(s,-s, s),(s,s, s),(-s,s, s)]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        dot.stroke((0, 255, 255))
        for i, j in edges:
            dot.line_3d(v[i], v[j])

    def draw(self):
        dot.background(dot.black)
        with dot.transform():
            dot.rotate(dot.frames * 0.02, 0, 1, 0)
            dot.rotate(dot.frames * 0.01, 1, 0, 0)
            dot.set_stroke_weight(1)
            self._draw_cube(2)
            dot.rotate(dot.frames * 0.05, 0, 1, 0)
            dot.rotate(dot.frames * 0.05, 1, 0, 0)
            dot.stroke((255, 255, 0))
            dot.set_stroke_weight(2)
            # polyline_3d draws a connected path through 3D points
            dot.polyline_3d(self.helix)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
