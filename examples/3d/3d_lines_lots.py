"""Infinite tunnel of wireframe cubes flying towards the camera."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.camera_3d()
        self.s       = 10    # cube spacing
        self.n       = 8     # grid size
        self.z       = 0.0
        self.speed   = 1
        self.dir_x   = 0.02
        self.dir_y   = 0.02
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.cam_pos = (3, 10, -65)
        dot.set_camera(self.cam_pos, (0, 0, 0))
        self._reset_weights()

    def _reset_weights(self):
        self.thick = np.ones((self.n, self.n, self.n + 1))
        mask = np.random.random(self.thick.shape) > 0.95
        self.thick[mask] = 10

    def _draw_cube(self, size, offset=(0, 0, 0)):
        s = size / 2
        ox, oy, oz = offset
        v = [(-s+ox,-s+oy,-s+oz),(s+ox,-s+oy,-s+oz),(s+ox,s+oy,-s+oz),(-s+ox,s+oy,-s+oz),
             (-s+ox,-s+oy, s+oz),(s+ox,-s+oy, s+oz),(s+ox,s+oy, s+oz),(-s+ox,s+oy, s+oz)]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        dot.stroke((0, 255, 255))
        for i, j in edges:
            dot.line_3d(v[i], v[j])

    def draw(self):
        dot.background((0, 0, 0))
        self.z       -= self.speed
        self.angle_x += self.dir_x
        self.angle_y += self.dir_y
        if np.random.random() > 0.98: self.dir_y *= -1
        if np.random.random() > 0.98: self.dir_x *= -1

        lx = 8  * np.cos(self.angle_x)
        ly = 10 * np.sin(self.angle_y)
        dot.set_camera(self.cam_pos, (lx, ly, 0))

        # Wrap z so the tunnel loops seamlessly
        if self.z <= self.s * 7:
            self.z += self.s * 7
            self._reset_weights()

        with dot.transform():
            off   = (self.s * self.n) / 2
            total = self.s * self.n
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n + 1):
                        dot.set_stroke_weight(self.thick[i, j, k])
                        x = i * self.s - off
                        y = j * self.s - off
                        z = k * self.s - off - self.z
                        while z < -total / 2: z += total
                        while z >  total / 2: z -= total
                        self._draw_cube(self.s, (x, y, z))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
