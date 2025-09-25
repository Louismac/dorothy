from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
                
    def draw(self):        
        dot.fill(dot.pink)
        for bin_num, bin_val in enumerate(dot.music.fft()[:100:8]):
            dot.circle((bin_num*60, int(bin_val*50)), 50)

        #Cover with a new alpha layer (instead of fully clearing with background)   
        dot.fill(dot.black)
        cover=dot.get_layer()
        dot.rectangle((0,0),(dot.width, dot.height),layer=cover)
        dot.draw_layer(cover,0.1)
        

MySketch()          







