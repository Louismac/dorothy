"""3D particle system: spawn spheres at the mouse position, fade over time."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(800, 800)

class Particle:
    def __init__(self, x, y, z):
        self.pos   = np.array([x, y, z], dtype=float)
        angle_xz   = np.random.random() * np.pi * 2
        angle_y    = (np.random.random() - 0.5) * np.pi
        speed      = 0.3
        self.vel   = np.array([np.cos(angle_xz) * np.cos(angle_y) * speed,
                               np.sin(angle_y) * speed,
                               np.sin(angle_xz) * np.cos(angle_y) * speed])
        self.size  = 1 + np.random.random() * 20
        self.life  = 1.0
        self.decay = np.random.random() * 0.005 + 0.001

    def update(self):
        self.pos  += self.vel
        self.life -= self.decay

    def draw(self):
        a = self.life * 2
        dot.fill((255 * a, 0, 255, 255 * a))
        dot.sphere(1 + self.size * self.life * dot.music.amplitude() * 5,
                   tuple(self.pos))

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/drums.wav")
        dot.camera_3d()
        self.particles = []
        self.scale_lfo = dot.get_lfo(freq=0.1, range=(50, 90))
        self.spin_lfo  = dot.get_lfo(freq=0.3, range=(0,  np.pi))

    def draw(self):
        dot.background((0, 0, 0, 255))
        z   = 1 + dot.lfo_value(self.scale_lfo)
        ang = dot.lfo_value(self.spin_lfo)
        dot.set_camera((np.cos(ang) * 50, np.sin(ang) * 50, z), (0, 0, 0))

        if np.random.random() > 0.75:
            px = 10 - (dot.mouse_x / dot.width)  * 20
            py = 10 - (dot.mouse_y / dot.height) * 20
            for _ in range(20):
                self.particles.append(Particle(px, py, 0))

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            p.draw()

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
