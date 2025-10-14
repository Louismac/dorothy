from dorothy import Dorothy

dot = Dorothy(width=800, height=600, title="Dorothy Demo")
    
# Simple test to verify basic drawing
class SimpleTest:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("Setup called!")
        print("Move mouse around the window to test mouse events")
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512,buffer_size=2048)
        # Camera is 2D by default
    
    def draw(self):
        # Clear background
        dot.background((50, 50, 60))
        
        # Draw a red circle
        dot.fill((255, 0, 0))
        dot.no_stroke()
        dot.circle((400, 300), 100)
        
        # Draw a blue rectangle with stroke
        dot.fill((0, 0, 255))
        dot.stroke((255, 255, 0))
        dot.set_stroke_weight(3)
        dot.rectangle((100, 100), (200, 200))
        
        # Draw a line
        dot.stroke((0, 255, 0))
        dot.set_stroke_weight(5)
        dot.line((50, 50), (750, 550))
        
        # Draw circle at mouse position
        dot.fill((255, 255, 255))
        dot.no_stroke()
        dot.circle((dot.mouse_x, dot.mouse_y), 20)
        
        # Print mouse position to console (throttled by frame rate)
        if dot.frames % 30 == 0:  # Print every 30 frames
            print(f"Mouse: ({dot.mouse_x}, {dot.mouse_y}), Down: {dot.mouse_down}")

SimpleTest()