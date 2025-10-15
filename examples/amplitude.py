from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Play file from your computer
        file_path = "../audio/gospel.wav"
        dot.music.start_file_stream(file_path,buffer_size=2048)
        
    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.rectangle((0,0),(dot.width,int(dot.music.amplitude()*dot.height*10)))

MySketch()   
    







