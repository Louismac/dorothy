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
        dot.set_camera((5, 5, 5), (0, 0, 0))
        dot.set_stroke_weight(10)
    
    def draw_cube(self,size):
        s = size / 2
        vertices = [
            (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
            (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
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
        
        with dot.transform():
            dot.rotate(dot.frames * 0.02, 0, 1, 0)  # Rotate around Y
            dot.rotate(dot.frames * 0.01, 1, 0, 0)  # Rotate around X
        
            # Draw cube edges
            self.draw_cube(2)

            dot.rotate(dot.frames * 0.05, 0, 1, 0)  # Rotate around Y
            dot.rotate(dot.frames * 0.05, 1, 0, 0)  # Rotate around X

            points = []
            for i in range(100):
                angle = i * 0.2
                radius = 1
                x = np.cos(angle) * radius
                z = np.sin(angle) * radius
                y = i / 100
                points.append((x, y, z))

            dot.stroke((255, 255, 0))
            dot.polyline_3d(points)

    
        
Example3D()