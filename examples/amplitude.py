from cv2 import rectangle
from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Play file from your computer
        file_path = "../audio/drums.wav"
        dot.music.start_file_stream(file_path)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        # dot.music.start_device_stream(2)
        
    def draw(self):
        dot.background(dot.white)
        rectangle(dot.canvas, (0,0),(dot.width,int(dot.music.amplitude()*dot.height*10)),dot.red,-1)

MySketch()   
    







