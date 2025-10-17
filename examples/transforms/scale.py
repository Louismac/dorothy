from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path)

        
    def draw(self):
        
        dot.background((77, 72, 79))
        factor = dot.music.amplitude() * 15 
        dot.stroke(dot.red)
        top_left = (dot.width//4, dot.height//4)
        bottom_right = (dot.width//4*3, dot.height//4*3)
        centre = np.array([dot.width//2, dot.height//2])

        with dot.transform():
        
            # move to middle to scale from centre as origin
            dot.translate(centre[0],centre[1])
            dot.scale(factor)
            # move back to top corner to draw
            dot.translate(-centre[0],-centre[1])
            dot.line(top_left, bottom_right)
            dot.circle(centre, 100)

        
MySketch()          







