from dorothy import Dorothy 
import numpy as np

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        dot.camera_3d()
        dot.set_camera((20, 20, 20), (0, 0, 0))
        # Load mesh
        self.tree = dot.load_obj("model/Tree1.obj")
        self.texture_layer = dot.get_layer()

    def draw(self):
        dot.background(dot.black)
        # Draw dynamic texture
        with dot.layer(self.texture_layer):
            dot.camera_2d()
            dot.background((50, 50, 100,10))
            dot.fill((255, 200, 100))
            x = (dot.frames * 5) % dot.width
            dot.circle((x, dot.height//2), 50)
        
        # Draw mesh
        with dot.transform():
            dot.camera_3d()
            dot.rotate(dot.frames * 0.01, 0, 1, 0)
            dot.fill((200, 150, 100))
            dot.draw_mesh(self.tree, self.texture_layer)

Example3D()