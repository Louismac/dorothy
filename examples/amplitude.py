from cv2 import rectangle
from Dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        print("setup")
        file_path = "../audio/hiphop.wav"
        dot.music.load_file(file_path, fft_size = 1024, buffer_size = 1024)
        dot.music.play()
        
    def draw(self):
        dot.background((0,0,0))
        rectangle(dot.canvas, (0,0),(dot.width,int(dot.music.amplitude*dot.height*10)),(0,255,0),-1)

MySketch()   
    







