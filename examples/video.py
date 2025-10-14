import cv2
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        self.player = cv2.VideoCapture("../images/louis.mov")
        self.cutout_layer = None
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        #Play file from your computer
        file_path = "../audio/disco.wav"
        dot.music.start_file_stream(file_path, fft_size=512)

        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        # print(sd.query_devices())
        #dot.music.start_device_stream(3)
        self.cutout_layer = dot.get_layer()

    def draw(self):
        success, camera_feed = self.player.read()
        if success:

            target_size = (640, 480)
            camera_feed = cv2.resize(camera_feed, target_size)
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            
            # Crop center quarter
            w, h = target_size
            crop_x1 = w // 4
            crop_y1 = h // 4
            crop_x2 = w // 4 * 3
            crop_y2 = h // 4 * 3
            cropped = camera_feed[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Calculate scaled size based on amplitude
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
        # Resize and convert
            dot.begin_layer(self.cutout_layer)
            dot.push_matrix()

            # This will now work!
            dot.translate(dot.width//2, dot.height//2, 0)
            factor = (dot.music.amplitude() * 5) + 1
            dot.scale(factor)
            dot.translate(-crop_w//2, -crop_h//2, 0)

            dot.paste(cropped, (0, 0))

            dot.pop_matrix()
            dot.end_layer()

            dot.draw_layer(self.cutout_layer)

        else:
            #LOOOOOOP
            print('no video')
            self.player.set(cv2.CAP_PROP_POS_FRAMES, 0)

MySketch()







