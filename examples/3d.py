from dorothy import Dorothy 
import librosa
import numpy as np

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("3D Setup!")
        dot.camera_3d()
        dot.set_camera((0, 0, 5), (0, 0, 0))
    
    def draw(self):
        dot.background((30, 30, 40))
        
        # Draw 3D rotating cube
        dot.push_matrix()
        dot.translate(0, 0, 0)
        dot.rotate(self.angle, 1, 1, 0)
        dot.fill((255, 100, 100))
        dot.box(1, 1, 1)
        dot.pop_matrix()
        
        # Draw 3D sphere
        dot.push_matrix()
        dot.translate(-2, 0, 0)
        dot.fill((100, 255, 100))
        dot.sphere(0.5)
        dot.pop_matrix()
        
        self.angle += 0.01
Example3D()