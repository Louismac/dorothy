import numpy as np
from cv2 import rectangle
from dorothy import Dorothy
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
        
        rave_id = dot.music.start_rave_stream("models/taylor_vocals.ts", latent_dim=latent_dim)
        #Explicitly set output device if you are using blackhole to direct audio as
        #a RAVE input (e.g. set this to your speakers to you can hear the output of RAVE)
        # rave_id = dot.music.start_rave_stream("models/taylor.ts", latent_dim=latent_dim, output_device = 1)

        ########## RANDOM ########## 
        #z = torch.randn((1,latent_dim,1))
        #dot.music.audio_outputs[rave_id].current_latent = z 
        
        ########## RUN FROM INPUT DEVICE ########## 
        #pass in the number of the device you want to input to RAVE e.g. blackhole or mic
        device_id = dot.music.start_device_stream(3)
        dot.music.update_rave_from_stream(device_id)

        ########## RUN FROM FILE ########## 
        # device_id = dot.music.start_file_stream("../audio/Wiley.wav")
        # set as input to rave (this mutes the source stream, use .gain to hear both)
        # dot.music.update_rave_from_stream(device_id)

        ########## CONSTANT Z BIAS ########## 
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

        ########## UPDATE BIAS FROM CALLBACK (EVERY BUFFER) ########## 

        # def sine_bias(frame_number, frequency=1, amplitude=1.0, phase=0, sample_rate=44100):
        #     t = frame_number / sample_rate
        #     value = amplitude * math.sin(2 * math.pi * frequency * t + phase)
        #     return value
        
        # self.ptr = 0
        # target = [1,2,4,8,16,32,64,128,256]
        # self.t = target[np.random.randint(len(target))]
        # def on_new_frame(buffer=np.zeros(2048)):
        #     n= len(buffer)
        #     # #Update a new random 
        #     dot.music.audio_outputs[0].z_bias = torch.randn(1,latent_dim,1)*0.1
        #     if self.ptr > (self.t*2048):
        #         self.t = target[np.random.randint(len(target))]
        #         z = torch.randn((1,latent_dim,1))
        #         dot.music.audio_outputs[rave_id].current_latent = z 
        #         print("new z!", self.t)
        #         self.ptr = 0
        #     # #OR
        #     # #update with oscilating bias
        #     # val = sine_bias(self.ptr, 0.4, 0.2)
        #     # dot.music.audio_outputs[0].z_bias = torch.tensor([val for n in range(latent_dim)]).reshape((1,latent_dim,1))

        #     self.ptr += n
        # dot.music.audio_outputs[rave_id].on_new_frame = on_new_frame

    def draw(self):
        dot.background((255,255,255))

        for bin_num, bin_val in enumerate(dot.music.fft()[::8]):
            bin_val = bin_val * 10
            pt1 = (bin_num*50, dot.height)
            pt2 = (0, dot.height-int(bin_val*1000))
            color = (0,(1-bin_val)*255,0)
            thickness = 1+int(bin_val*2)
            dot.stroke(color)
            dot.set_stroke_weight(thickness)
            dot.line(pt1, pt2)

MySketch()          







