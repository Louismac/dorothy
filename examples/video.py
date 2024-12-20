import cv2
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        self.player = cv2.VideoCapture("../images/louis.mov")
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)
        
        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
            
    def draw(self):
        success, camera_feed = self.player.read()
        if success:
            #resize and color camera
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            camera_feed = cv2.resize(camera_feed,(dot.width, dot.height))
            
            #cut out centre
            new_layer = dot.get_layer()
            top_left = (dot.width//4, dot.height//4)
            bottom_right = (dot.width//4*3, dot.height//4*3)
            dot.paste(new_layer, camera_feed[top_left[0]:bottom_right[1],top_left[1]:bottom_right[1]], top_left)
            
            #scale to amplitude
            factor = (dot.music.amplitude() * 5) + 1
            origin = (dot.width//2,dot.height//2)
            new_layer = dot.scale_layer(new_layer, factor, factor, origin)
            
            dot.canvas = camera_feed
            dot.draw_layer(new_layer)
        else:
            #LOOOOOOP
            print('no video')
            self.player.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
MySketch()          







