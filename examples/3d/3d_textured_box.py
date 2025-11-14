from dorothy import Dorothy 
import numpy as np
from PIL import Image

dot = Dorothy(640,480)
class Example3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        # Create 6 different layers
        self.front_layer = dot.get_layer()
        self.back_layer = dot.get_layer()
        self.right_layer = dot.get_layer()
        self.left_layer = dot.get_layer()
        self.top_layer = dot.get_layer()
        self.bottom_layer = dot.get_layer()
        dot.camera_3d()
        dot.set_camera((10, 10, 10), (0, 0, 0))
        self.mario = Image.open('../images/space.jpg')

    def draw(self):
        
        dot.camera_2d()
        # Draw different content to each layer
        with dot.layer(self.front_layer):
            dot.paste(self.mario, (0,0))
            
        with dot.layer(self.back_layer):
            dot.fill(dot.blue)
            dot.circle((dot.frames%dot.width, dot.frames%dot.height), 2)
            
        with dot.layer(self.right_layer):
            dot.fill(dot.yellow)
            dot.circle((dot.frames%dot.width, dot.frames%dot.height), 10)
            
        with dot.layer(self.left_layer):
            dot.fill(dot.red)
            dot.circle((dot.frames%dot.width, dot.frames%dot.height), 20)
            
        with dot.layer(self.top_layer):
            dot.fill(dot.magenta)
            dot.circle((dot.frames%dot.width, dot.frames%dot.height), 30)
            
        with dot.layer(self.bottom_layer):
            dot.fill(dot.cyan)
            dot.circle((dot.frames%dot.width, dot.frames%dot.height), 20)
        
        # Draw box with different textures per face
        dot.background((10, 10, 20))
        dot.camera_3d()
        with dot.transform():
            dot.rotate(dot.frames * 0.01, 0, 1, 0)
            dot.rotate(dot.frames * 0.005, 1, 0, 0)
            
            dot.box((2, 2, 2), texture_layers={
                'front': self.front_layer,
                'back': self.back_layer,
                'right': self.right_layer,
                'left': self.left_layer,
                'top': self.top_layer,
                'bottom': self.bottom_layer
            })
            angle = dot.frames * 0.05
            x = 10 * np.cos(angle)
            z = 10 * np.sin(angle)
            dot.box((1, 1, 1),(x,1,z), texture_layers=self.front_layer)
        
Example3D()