from dorothy import Dorothy 
import numpy as np

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        dot.camera_3d()
        dot.set_camera((40, 40, 40), (0, 0, 0))
        # Load mesh
        self.model = dot.load_obj("model/cottage_obj.obj")
        #self.model = dot.load_obj("model/Tree1.obj")
        self.texture_layer = dot.get_layer()

    def draw(self):
        dot.background(dot.black)
        # Draw dynamic texture
        with dot.layer(self.texture_layer):
            dot.camera_2d()
            dot.fill((255, 0, 0, 100))
            dot.rectangle((0,0),(dot.width, dot.height-2))
            dot.fill((255, 200, 100))
            x = (dot.frames * 5) % dot.width
            dot.circle((x, dot.height//2), 50)
        
        # Draw mesh
        with dot.transform():
            dot.camera_3d()
            dot.rotate(dot.frames * 0.005, 0, 1, 0)
            # fill with colour
            dot.fill((0,0,255,50))
            dot.draw_mesh(self.model)
            #or texture with a layer
            dot.translate(0,20,0)
            dot.draw_mesh(self.model, self.texture_layer)

            

Example3D()