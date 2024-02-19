from Dorothy import Dorothy
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
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        
        dot.music.play()

    def draw(self):
        if dot.music.is_beat():
            self.current_image = np.random.randint(len(self.dataset))
       
        to_paste = self.dataset[self.current_image]
        #Paste into canvas
        dot.paste(dot.canvas, to_paste)

MySketch()          







