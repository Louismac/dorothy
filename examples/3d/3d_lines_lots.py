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
        self.camera_pos = -45
        self.s = 10  # spacing between cubes
        self.cube_size = 7  # grid dimensions
        self.z = 0
        self.speed = 1  # movement speed
        self.dir_x = 0.02
        self.angle_x = 0
        self.dir_y = 0.02
        self.angle_y = 0
        dot.set_camera((1, -2, self.camera_pos), (0, 0, 0))

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
        dot.background((0, 0, 0))
        dot.set_stroke_weight(1)
        
        # Update position
        self.z -= self.speed
        self.angle_x += self.dir_x
        self.angle_y += self.dir_y
        if np.random.random()>0.98:
            self.dir_y = -self.dir_y
        if np.random.random()>0.98:
            self.dir_x = -self.dir_x
        x = 8 * np.cos(self.angle_x)
        y = 6 * np.sin(self.angle_y)
        dot.set_camera((1, -2, self.camera_pos),(x, y, 0))
        
        # Wrap z when it exceeds one cube spacing
        if self.z <= self.s*4:
            self.z += self.s
        
        with dot.transform():
            offset = (self.s * self.cube_size) / 2
            
            # Draw grid of cubes with wrapping in Z
            for i in range(self.cube_size):
                for j in range(self.cube_size):
                    for k in range(self.cube_size + 1):  # Extra layer for seamless wrapping
                        dot.set_stroke_weight(1 if np.random.random()<0.97 else 2)
                        # Calculate base position
                        x = (i * self.s) - offset
                        y = (j * self.s) - offset
                        z = (k * self.s) - offset - self.z
                        
                        # Wrap z coordinate to create infinite tunnel
                        # Keep cubes within visible range
                        total_depth = self.s * self.cube_size
                        while z < -total_depth / 2:
                            z += total_depth
                        while z > total_depth / 2:
                            z -= total_depth
                        
                        self.draw_cube(self.s, (x, y, z))

Example3D()