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
        latent_dim = 128
        
        rave_id = dot.music.start_rave_stream(
            "models/drum_loops/", 
            latent_dim=latent_dim, output_device=4, buffer_size = 1024)
        
        # rave_id = dot.music.start_rave_stream(
        #     "models/rave_gospel.ts",
        #     buffer_size = 4096, 
        #     latent_dim=latent_dim,output_device=5)
        dot.music.audio_outputs[rave_id].load_cluster_results("models/")
        layer_names = list(dot.music.audio_outputs[rave_id].cluster_results.keys())
        device_id = dot.music.start_device_stream(3, buffer_size = 44100)        
        #device_id = dot.music.start_file_stream("../audio/Wiley.wav",buffer_size=2048)
        dot.music.update_rave_from_stream(device_id)
        n_layers = len(layer_names)
        clusters = 6
        print(n_layers*clusters)
        w = dot.width//n_layers
        h = dot.height//clusters
        for i in range(n_layers):
            for j in range(clusters):
                def on_release(btn):
                    vals = btn.id.split("_")
                    c = int(vals[1])
                    l = layer_names[int(vals[0])]
                    f = vals[2]
                    print(c,l,f)
                    is_on = dot.music.audio_outputs[rave_id].toggle_bend(f,l,c)
                    if f == "noise":
                        btn.set_style(idle_color = dot.red if is_on else dot.white)
                    else:
                        btn.set_style(idle_color = dot.red if is_on else dot.black)
                btn = dot.create_button(i*w, j*h, w, h//2, 
                        id=f"{i}_{j}_noise", on_release=on_release)
                btn.set_style(idle_color = dot.white)
                btn = dot.create_button(i*w, j*h + h//2, w, h//2, 
                        id=f"{i}_{j}_lfo", on_release=on_release)
                btn.set_style(idle_color = dot.black)
        self.prev = dot.millis

    def draw(self):
        dot.background((40, 40, 50))
        
        # Update and draw buttons
        dot.update_buttons()
        dot.draw_buttons()

MySketch()          







