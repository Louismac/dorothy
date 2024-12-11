from dorothy import Dorothy 
from cv2 import rectangle
import math

dot = Dorothy() 
class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        dot.music.start_file_stream("../../audio/drums.wav")
        dot.music.pause()

    def draw(self):
        dot.background(dot.white)
        seg = math.floor((dot.mouse_x/dot.width)*5)
        print(seg)
        if seg==2:
            rectangle(dot.canvas, (dot.width//2-100,0),(dot.width//2+100, dot.height), dot.green, -1)
            dot.music.resume()
        else:
            dot.music.pause()
MySketch() 