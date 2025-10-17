from dorothy import Dorothy
import numpy as np

dot = Dorothy(800,600)

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        #Play file from your computer
        file_path = "../audio/drums.wav"
        dot.music.start_file_stream(file_path)
        #make a sine wave
        freq = 10
        num_pts = 50
        self.x = np.linspace(1,dot.width,num_pts)
        self.y = (np.sin(np.linspace(0,np.pi*freq,num_pts))*200)

    def draw(self):

        dot.background(dot.white)

        #amplify y values by volume
        y = np.array(self.y * (0+(dot.music.amplitude()))*2).astype(np.int32)+300

        # draw dots
        dot.fill(dot.blue)
        dot.no_stroke()
        for pt in zip(self.x,y):
            dot.circle(pt, 2)

        # fit line    
        z = np.polyfit(self.x, y, 10)
        resolution = 150
        draw_x = np.linspace(0, dot.width, resolution)
        draw_y = np.polyval(z, draw_x) 
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        
        # draw polyline
        dot.stroke(dot.green)
        dot.set_stroke_weight(2)
        dot.no_fill()
        dot.polyline(draw_points, closed=False)

        # Concave polygon (pac-man shape)
        dot.fill(dot.yellow)
        dot.polygon([
            (100, 200), (150, 150), (200, 200),
            (200, 300), (150, 250)
        ])
        
        # Star shape (concave)
        points = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            r = 100 if i % 2 == 0 else 50
            x = 400 + r * np.cos(angle)
            y = 300 + r * np.sin(angle)
            points.append((x, y))
        
        dot.fill(dot.red)
        dot.stroke(dot.white)
        dot.set_stroke_weight(2)
        dot.polygon(points)
    
MySketch()





