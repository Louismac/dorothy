"""Orbit the camera around the scene using trigonometry."""
from dorothy import Dorothy
import numpy as np
import math

dot = Dorothy(1080, 960)

class MySketch:

    def setup(self):
        dot.camera_3d()
        dot.set_camera((0, 0, 5), (0, 0, 0))

    def draw(self):
        dot.background((30, 30, 40))
        a = dot.frames * 0.01
        # Move camera in a circle on the XZ plane while looking at origin
        dot.set_camera((5 * math.cos(a), 2, 5 * math.sin(a)), (0, 0, 0))
        dot.camera_3d()
        with dot.transform():
            dot.fill((100, 100, 255))
            dot.box((1, 1, 1), (np.sin(np.pi * dot.frames * 0.01) * 5, 0, 0))
        dot.fill((255, 100, 100))
        dot.sphere(0.5, (1, 0, 0))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
