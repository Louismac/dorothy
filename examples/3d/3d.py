from dorothy import Dorothy 
dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.camera_3d()
        dot.set_camera((0, 0, 5), (0, 0, 0))
    
    def draw(self):
        dot.fill(dot.red)
        dot.sphere(0.5, (-0.5, 0.5, 0.5))
        dot.fill(dot.blue)
        dot.box((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     
Example3D()