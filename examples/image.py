from dorothy import Dorothy 
import librosa
import numpy as np
from PIL import Image


dot = Dorothy(640,480)

class MySketch:
    def __init__(self):
        self.angle = 0
        self.image = None
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        self.rgb_image = Image.open('../images/space.jpg')
        self.mario = Image.open('../images/mario.png')
        self.grayscale = self.rgb_image.convert('L')
        self.grayscale = np.array(self.grayscale)
        self.rgb_image = np.array(self.rgb_image)


    def draw(self):
        dot.background((30, 30, 40))
        
        # Paste image at different positions with different effects
        
        # Static image
        dot.paste(self.rgb_image, (50, 50))
        
        # Rotated position (using angle)
        x = 400 + 150 * np.cos(self.angle)
        y = 300 + 150 * np.sin(self.angle)
        dot.paste(self.mario, (int(x), int(y)), size=(50, 50), alpha=0.8)
        
        # Scaled and faded
        scale = 0.5 + 0.5 * np.sin(self.angle * 2)
        size = int(100 * scale)
        dot.paste(self.rgb_image, (0, 0), size=(size, size), alpha=scale)
        
        self.angle += 0.02
        
MySketch()