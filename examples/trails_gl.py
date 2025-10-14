from dorothy import Dorothy 
import librosa
import numpy as np

dot = Dorothy(640,480)

class MySketch:
    def __init__(self):
        self.layer1 = None
        self.layer2 = None
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("=== Layer Transparency Test ===")
        self.layer1 = dot.get_layer()
        self.layer2 = dot.get_layer()
        
        # Draw static content to layer 1 (red circle)
        dot.begin_layer(self.layer1)
        dot.fill((255, 0, 0))
        dot.no_stroke()
        dot.circle((300, 300), 100)
        dot.end_layer()
        
        self.ptr = 0
        
        print("Layers created and drawn to")
    
    def draw(self):
        # Clear background to dark gray
        self.ptr += 1
        dot.background((180, 250, 100))
         # Draw static content to layer 2 (blue circle)
        dot.begin_layer(self.layer2)
        dot.fill((0, 0, 255))
        dot.no_stroke()
        dot.circle((self.ptr%dot.width, 300), 100)
        dot.end_layer()

        # Draw static content to layer 1 (red circle)
        dot.begin_layer(self.layer1)
        dot.fill((255, 0, 0))
        dot.no_stroke()
        dot.circle(((self.ptr*2)%dot.width, (self.ptr*2)%dot.width), 100)
        dot.end_layer()
        
        # Draw both layers with different alpha values
        dot.draw_layer(self.layer1, alpha=1.0)  # Full opacity
        dot.draw_layer(self.layer2, alpha=0.5)  # Half transparent
        
        # Draw some direct shapes on screen for comparison
        dot.fill((0, 255, 0))
        dot.no_stroke()
        dot.circle((400, 150), 50)  # Green circle on screen
        
        if dot.frames == 1:
            print("You should see:")
            print("- Red circle (opaque) at left")
            print("- Blue circle (50% transparent) at right - should see gray through it")
            print("- Green circle at top")
    
MySketch()