import numpy as np
from cv2 import rectangle
from src.Dorothy import Dorothy
import sounddevice as sd
import torch 
import math

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        #Output RAVE from speakers
        latent_dim = 8
        print(sd.query_devices())
        
        rave_id = dot.music.start_rave_stream("models/taylor.ts", latent_dim=latent_dim, output_device=1)
        #Explicitly set output device if you are using blackhole to direct audio as
        #a RAVE input (e.g. set this to your speakers to you can hear the output of RAVE)
        # rave_id = dot.music.start_rave_stream("models/taylor.ts", latent_dim=latent_dim, output_device = 1)

        #Random
        z = torch.randn((1,latent_dim,1))
        dot.music.update_rave_latent(z) 
        
        #start stream, pass in the number of the device you want to input to RAVE e.g. blackhole
        # device_id = dot.music.start_device_stream(2)
        # dot.music.update_rave_from_stream(device_id)

        # start file stream (to be used as input to RAVE)
        #device_id = dot.music.start_file_stream("../audio/Wiley.wav")
        # set as input to rave (this mutes the source stream, use .gain to hear both)
        #dot.music.update_rave_from_stream(device_id)

        # d0 = 1.09  # change in latent dimension 0
        # d1 = -3 
        # d2 = 0.02
        # d3 = 0.5 
        # z_bias = torch.zeros(1, latent_dim, 1)
        # z_bias[:, 0] = torch.linspace(d0, d0, z_bias.shape[-1])
        # z_bias[:, 1] = torch.linspace(d1, d1, z_bias.shape[-1])
        # z_bias[:, 2] = torch.linspace(d2, d2, z_bias.shape[-1])
        # z_bias[:, 3] = torch.linspace(d3, d3, z_bias.shape[-1])
        #Constant bias
        #dot.music.audio_outputs[0].z_bias = z_bias

        # def sine_bias(frame_number, frequency=1, amplitude=1.0, phase=0, sample_rate=44100):
        #     t = frame_number / sample_rate
        #     value = amplitude * math.sin(2 * math.pi * frequency * t + phase)
        #     return value
        
        self.ptr = 0
        # target = [1,2,4,8,16,32,64,128,256]
        # self.t = target[np.random.randint(len(target))]
        def on_new_frame(buffer=np.zeros(2048)):
            n= len(buffer)
            # #Update a new random 
            # # dot.music.audio_outputs[0].z_bias = torch.randn(1,latent_dim,1)*0.1
            # if self.ptr > (self.t*2048):
            #     self.t = target[np.random.randint(len(target))]
            #     z = torch.randn((1,latent_dim,1))
            #     dot.music.update_rave_latent(z) 
            #     print("new z!", self.t)
            #     self.ptr = 0
            # #ORconda
            # #update with oscilating bias
            # val = sine_bias(self.ptr, 0.4, 0.2)
            # dot.music.audio_outputs[0].z_bias = torch.tensor([val for n in range(latent_dim)]).reshape((1,latent_dim,1))

            self.ptr += n
        dot.music.audio_outputs[rave_id].on_new_frame = on_new_frame
        dot.music.play()
        
    def draw(self):
        dot.background((255,255,255))
        win_size = 10
        scale = 15
        alpha = 0.4
        #Only draw 20 rectangles
        for i in range(20):
            #Get max fft val in window of frequeny bins
            window = dot.music.fft_vals[i*win_size:(i+1)*win_size]
            val = int(np.mean(window))
            width = val*(i*scale)
            top_left = (dot.width//2-width,dot.height//2-width)
            bottom_right = (dot.width//2+width,dot.height//2+width)
            #draw to an alpha layer
            new_layer = dot.to_alpha(alpha)
            rectangle(new_layer, top_left, bottom_right, (10*val,26*val,143*val), -1)
        #Call this when you want to render the alpha layers to the canvas (e.g. to draw something else on top of them)
        dot.update_canvas()
        # top_left = (dot.width//2-10,dot.height//2-10)
        # bottom_right = (dot.width//2+10,dot.height//2+10)
        # rectangle(dot.canvas, top_left, bottom_right, (255,255,255), -1)

MySketch()          







