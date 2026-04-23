"""Load an OBJ mesh and apply a dynamic layer as its texture."""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(640, 480)

class MySketch:

    def setup(self):
        dot.camera_3d()
        dot.set_camera((40, 40, 40), (0, 0, 0))
        self.model         = dot.load_obj("models/cottage_obj.obj")
        self.texture_layer = dot.get_layer()

    def draw(self):
        dot.background(dot.black)

        # Redraw the texture layer each frame
        with dot.layer(self.texture_layer):
            dot.camera_2d()
            dot.fill((255, 0, 0, 100))
            dot.rectangle((0, 0), (dot.width, dot.height - 2))
            dot.fill((255, 200, 100))
            dot.circle(((dot.frames * 5) % dot.width, dot.height // 2), 50)

        with dot.transform():
            dot.camera_3d()
            dot.translate(0, -10, 0)
            dot.rotate(dot.frames * 0.005, 0, 1, 0)
            dot.fill((0, 0, 255, 50))
            dot.draw_mesh(self.model)                      # flat colour
            dot.translate(0, 20, 0)
            dot.draw_mesh(self.model, self.texture_layer)  # layer texture

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
