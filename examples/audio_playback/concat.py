

from dorothy import Dorothy
from dorothy.Audio.effects import Reverb

dot = Dorothy()

class MySketch:

    def setup(self):
        
        cat_idx = dot.music.start_concat_stream(
            "../audio",
            unit_size=120,      # 120 ms units
            n_candidates=8,     # pick randomly from top-8 matches
            features=('mfcc', 'centroid', 'rms','chroma'),
        )
       
        
        self.cat = dot.music.audio_outputs[cat_idx]
        self.cat_idx = cat_idx
        
        # Drive from audio file if you want
        # file_path = "../audio/disco.wav"
        # audio_file = dot.music.start_file_stream(file_path)
        # dot.music.drive_concat_from_stream(audio_file, self.cat_idx)

        #play forever
        self.cat.note_on(440, vel=0.8)
        self.verb = Reverb(wet=0.4)
        self.cat.add_effect(self.verb)
        

    def draw(self):
        dot.background(dot.darkblue)

        # Map mouse position to centroid (0–8000 Hz) and rms (0–0.8)
        self.cat.target['centroid'] = dot.mouse_x / dot.width * 8000
        self.cat.target['rms']      = (1.0 - dot.mouse_y / dot.height) * 0.8

        #Live code grain params if you want!
        self.cat.spread = 0.1
        self.cat.grain_size = 120
        self.cat.density = 12
        self.cat.n_grains = 30
        self.cat.attack = 0.1
        self.cat.decay = 0.1
        self.verb.room_size = 0.4
        
        amp = dot.music.amplitude(self.cat_idx)
        dot.fill(dot.white)
        dot.circle((dot.width / 2, dot.height / 2), amp * 300)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
