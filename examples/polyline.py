from dorothy import Dorothy
import numpy as np
import cv2

dot = Dorothy(1000,640)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Play file from your computer
        file_path = "../audio/drums.wav"
        dot.music.start_file_stream(file_path)
        #make a sine wave
        freq = 10
        self.x = np.linspace(1,1000,100)
        self.y = (np.sin(np.linspace(0,np.pi*freq,100))*200)
        

    def draw(self):

        dot.background(dot.white)

        #amplify y values by volume
        y = np.array(self.y * (0+(dot.music.amplitude()))*2).astype(np.int32)+300

        # draw dots
        dot.fill(dot.blue)
        dot.no_stroke()
        for pt in zip(self.x,y):
            dot.circle(pt,5)

        # fit line    
        z = np.polyfit(self.x, y, 10)
        draw_x = np.linspace(0, 1000, 100)
        draw_y = np.polyval(z, draw_x) 
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        
        # draw polyline
        dot.stroke(dot.green)
        dot.set_stroke_weight(2)
        dot.no_fill()
        dot.poly([draw_points])

MySketch()   
    







