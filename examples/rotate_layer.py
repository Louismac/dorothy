from dorothy import Dorothy
import numpy as np

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, buffer_size=2048)
        dot.fill((255, 0, 0))
        self.layer = dot.get_layer()
        dot.begin_layer(self.layer)
        dot.fill(dot.yellow)
        top_left = (dot.width//4, dot.height//4)
        bottom_right = (dot.width//4*3, dot.height//4*3)
        dot.rectangle(top_left, bottom_right)
        dot.end_layer()
        
        self.theta = 0

    def draw(self):
        # Clear with semi-transparent red for trails
        dot.fill((255, 0, 0, 20))
        dot.rectangle((0, 0), (dot.width, dot.height))
        
        # Rotate and draw the yellow rectangle
        self.theta += dot.music.amplitude() * np.pi
        centre = np.array([dot.width//2, dot.height//2])
        
        dot.push_matrix()
        dot.translate(centre[0], centre[1]) 
        dot.rotate(self.theta)
        dot.translate(-centre[0], -centre[1]) 
        dot.draw_layer(self.layer)
        dot.pop_matrix()

MySketch()          
