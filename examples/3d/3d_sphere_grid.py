"""Audio-reactive 3D sphere grid; camera spins with amplitude."""
from dorothy import Dorothy
import math

dot = Dorothy(1080, 960)

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/drums.wav")
        dot.camera_3d()
        self.theta = 0.0

    def draw(self):
        dot.background(dot.white)
        amp = dot.music.amplitude()
        self.theta += amp
        dot.set_camera((7 * math.cos(self.theta), 3, 7 * math.sin(self.theta)),
                       (0, 0, 0))
        dot.fill(dot.red)
        for i in range(10):
            for j in range(10):
                dot.sphere(amp * 2, (i - 5, j - 5, 0))
        dot.rgb_split(bake=False, offset=amp * 0.5)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
