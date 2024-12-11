from dorothy import Dorothy 
from cv2 import circle

dot = Dorothy() 
class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        print("setup")
        dot.music.start_file_stream("../../audio/meow.wav")

    def draw(self):
        dot.background(dot.white)
        print(dot.music.amplitude())
        #Cat detector!
        if dot.music.amplitude() > 0.01:
            circle(dot.canvas, (dot.width//2,dot.height//2),200, dot.black, -1)
        
MySketch() 