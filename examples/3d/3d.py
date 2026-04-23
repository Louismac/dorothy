"""Basic 3D: switch to 3D camera then draw a sphere and a box."""
from dorothy import Dorothy

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        # camera_3d() switches the renderer to perspective projection
        dot.camera_3d()
        dot.set_camera((0, 0, 5), (0, 0, 0))   # position, look-at

    def draw(self):
        dot.fill(dot.red)
        dot.sphere(0.5, (-0.5, 0.5, 0.5))      # radius, position
        dot.fill(dot.blue)
        dot.box((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # size, position

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
