from dorothy import Dorothy
import numpy as np

dot = Dorothy(300,300)

class MySketch:

    thumbnail_size = (dot.width, dot.height)
    dataset = dot.get_images("../images/aligned_faces", thumbnail_size)
    current_image = 0

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
                
    def draw(self):
        if dot.music.is_beat():
            self.current_image = np.random.randint(len(self.dataset))
       
        to_paste = self.dataset[self.current_image]
        #Paste into canvas
        dot.paste(to_paste, (0,0))

MySketch()          







