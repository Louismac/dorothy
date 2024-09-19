from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
        
    def setup(self):
        print("setup")
        file_path = "../audio/bass.wav"
        dot.music.start_file_stream(file_path)
        
    def draw(self):
        dot.background((255,255,255))
        waveform = dot.draw_waveform(dot.get_layer(), col=dot.black, with_playhead=True)
        dot.draw_layer(waveform)

MySketch()          