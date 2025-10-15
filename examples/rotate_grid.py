from cv2 import rectangle
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

        dot.stroke(dot.grey)
        
    def draw(self):
        
        dot.background((22, 208, 165))

        size = 40
        border = 10
        grid = 5
        x_offset = (dot.width - ((size+border)*grid)) //2
        y_offset = (dot.height - ((size+border)*grid)) //2
        for i in range(grid):
            for j in range(grid):
                #Where to draw the shape?
                x = i * (size+border) + x_offset
                y = j * (size+border) + y_offset
                #Draw to it
                top_left = (x,y)
                bottom_right = np.array([x+size, y+size])
                
                theta = dot.music.amplitude() * 15 * 2 * np.pi
                origin = np.array([x+size/2, y+size/2])
                dot.push_matrix()
                dot.translate(origin[0],origin[1])
                #rotate
                dot.rotate(theta)
                dot.translate(-origin[0],-origin[1])
                if i % 2 == 0:
                    dot.line(top_left, origin)
                else:
                    dot.rectangle(top_left, bottom_right)
                dot.pop_matrix()
                

MySketch()          







