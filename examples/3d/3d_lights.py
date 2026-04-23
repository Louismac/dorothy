"""3D lighting: an orbiting point light on two animated objects."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.camera_3d()
        dot.set_camera((5, 5, 5), (0, 0, 0))

    def draw(self):
        dot.camera_3d()
        dot.background((30, 30, 40))

        # Orbit the light source around the scene
        a = dot.frames * 0.01
        dot.renderer.light_pos = (5 * np.cos(a), 10 * np.sin(a / 2), 5 * np.sin(a))

        # Toggle lighting every 50 frames to compare lit vs flat shading
        if dot.frames % 50 == 0:
            dot.use_lighting(not dot.renderer.use_lighting)

        dot.fill((0, 0, 255, 120))
        dot.box((1, 1, 1), (np.sin(np.pi * dot.frames * 0.01), 0, 0))
        dot.fill((255, 0, 0, 200))
        dot.sphere(1, (0, np.sin(np.pi * dot.frames * 0.01), 0))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
