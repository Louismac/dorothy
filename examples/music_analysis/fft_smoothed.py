from dorothy import Dorothy
import sounddevice as sd

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  
            
    def setup(self): 

        #Play file from your computer
        file_path = "../audio/gospel.wav"
        fft_size = 512
        self.skip = 16
        dot.music.start_file_stream(file_path, fft_size=fft_size)
        #get object for running average
        self.window = dot.get_window(20, dims = fft_size//self.skip)
                
    def draw(self):
        
        dot.background(dot.black)

        #Multi dimensional moving average
        smoothed = self.window.add(dot.music.fft()[::self.skip])

        for bin_num, bin_val in enumerate(smoothed):
            bin_val = bin_val * 10
            pt1 = (bin_num*50, dot.height)
            pt2 = (0, dot.height-int(bin_val*1000))
            color = (0,(1-bin_val)*255,0)
            thickness = 1+int(bin_val*2)
            dot.stroke(color)
            dot.set_stroke_weight(thickness)
            dot.line(pt1, pt2)

MySketch()






