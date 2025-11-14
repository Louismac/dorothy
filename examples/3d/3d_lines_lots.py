from dorothy import Dorothy 
import numpy as np

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("3D Setup!")
        dot.camera_3d()
        dot.set_camera((50, 50, 50), (0, 0, 0))
        
    
    def draw_cube(self, size, offset=(0, 0, 0)):
        s = size / 2
        ox, oy, oz = offset
        
        vertices = [
            (-s + ox, -s + oy, -s + oz), (s + ox, -s + oy, -s + oz), 
            (s + ox, s + oy, -s + oz), (-s + ox, s + oy, -s + oz),
            (-s + ox, -s + oy, s + oz), (s + ox, -s + oy, s + oz), 
            (s + ox, s + oy, s + oz), (-s + ox, s + oy, s + oz),
        ]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        dot.stroke((0, 255, 255))
        for i, j in edges:
            dot.line_3d(vertices[i], vertices[j])

    def draw(self):
        dot.background(dot.black)
        dot.set_stroke_weight(1)
        s = 5
        with dot.transform():
            dot.rotate(dot.frames * 0.02, 0, 1, 0)  # Rotate around Y
            dot.rotate(dot.frames * 0.01, 1, 0, 0)  # Rotate around X
            cube_size = 6
            for i in range(cube_size):
                for j in range(cube_size):
                    for k in range(cube_size):
                        with dot.transform():
                            self.draw_cube(s,(i*s,j*s,k*s))

            

            

    
        
Example3D()