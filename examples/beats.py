
from Dorothy import Dorothy

dot = Dorothy()

class MySketch:

    show_beat = 0

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        print("setup")
        file_path = "../audio/hiphop.wav"
        dot.music.load_file(file_path, fft_size = 1024, buffer_size = 1024)
        dot.music.play()
        
    def draw(self):
        col = (0,0,0)
        if dot.music.is_beat():
            self.show_beat = 10
        
        if self.show_beat > 0:
            col = (255,255,255)
        
        dot.background(col)
        self.show_beat -= 1

MySketch()   
    







