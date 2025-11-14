from dorothy import Dorothy

dot = Dorothy(640,480)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
        
    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        dot.circle((dot.width//2,dot.height//2),int(dot.music.amplitude()*dot.height*2))

MySketch()   
    







