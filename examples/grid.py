import numpy as np
from cv2 import rectangle
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    palette = np.array([
    [107,45,92],
    [240,56,107],
    [255,83,118],
    [248,192,200]
    ])

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=1024)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        
        dot.music.play()
    
    def draw(self):
        if dot.frame%2==0:
            grid = 7
            win_size = 7
            size = dot.width//grid
            for i in range(grid**2):
                x = i % grid * size
                y = i // grid * size
                fft = dot.music.fft()
                window = np.max(fft[i*win_size:(i+1)*win_size])
                color = self.palette[i % len(self.palette)]
                color = color*window
                rectangle(dot.canvas,
                        (x,y),(x+size,y+size),
                        color,-1)

MySketch()      







