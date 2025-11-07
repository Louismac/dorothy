from dorothy import Dorothy
from dorothy.Audio import Sampler

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        #Output RAVE from speakers
        latent_dim = 128
        rave_id = dot.music.start_rave_stream(
            "/Users/lmccallum/Documents/checkpoints/RAVE/taylor_vocals_mono_e18d54798e/", 
            latent_dim=latent_dim, output_device=4, buffer_size = 1024)
        dot.music.audio_outputs[rave_id].load_cluster_results("audio_playback/")
        dot.music.audio_outputs[rave_id].pause()
        layer_names = list(dot.music.audio_outputs[rave_id].cluster_results.keys())
        layer_names = ["audioplayback/optimised/" + l for l in layer_names]
        n_layers = len(layer_names)
        paths = [[l+"/top_5_bpm_latent_maximised.wav",l+"/top_5_pitch_latent_maximised.wav"] for l in layer_names]
        self.sampler = Sampler(dot.music)
        self.sampler.load(paths)
        w = dot.width//n_layers
        h = dot.height//2
        for i in range(n_layers):
            def on_hover(btn):
                vals = btn.id.split("_")
                i = int(vals[0])
                o = 0 if vals[1] == "bpm" else 1
                self.sampler.trigger((i*2)+o)
            btn = dot.create_button(i*w, 0, w, h, 
                    id=f"{i}_bpm", on_hover=on_hover)
            btn.set_style(idle_color = dot.white)
            btn = dot.create_button(i*w, h, w, h, 
                    id=f"{i}_pitch", on_hover=on_hover)
            btn.set_style(idle_color = dot.black)
        self.prev = dot.millis

    def draw(self):
        dot.background((40, 40, 50))
        
        # Update and draw buttons
        dot.update_buttons()
        dot.draw_buttons()

MySketch()          







