import numpy as np
import librosa

class Sampler:
    def __init__(self, dot):
        #timing
        self.set_bpm(120)
        self.tick_ctr = 0
        self.next_tick = self.tick_length
        self.samples = [np.zeros(1024)]
        self.sample_pos = [-1 for _ in self.samples]
        self.sequence = np.zeros((len(self.samples),1))

        #Audio Callback function
        def get_frame(size):
            audio = np.zeros(size)
            for i,p in enumerate(self.sample_pos):
                if p >= 0:
                    end = p+size
                    if end >= len(self.samples[i]):
                        remaining = len(self.samples[i])-p
                        end = p + remaining
                        audio[:remaining] += self.samples[i][p:end]
                        self.sample_pos[i] = -1
                    else: 
                        audio += self.samples[i][p:end]
                        self.sample_pos[i] += size
            return audio
        
        dot.music.start_dsp_stream(get_frame, sr = 22050, buffer_size=128)

    def tick(self, time):
        is_tick = False
        #update sequencer pointer
        if time > self.next_tick:
            is_tick = True
            for i,s in enumerate(self.sequence):
                seq_ptr = self.tick_ctr%len(s)
                #trigger
                if s[seq_ptr] > 0:
                    self.sample_pos[i] = 0
            self.tick_ctr += 1
            self.next_tick += self.tick_length
        return is_tick

    def load(self, paths):
        self.samples = [librosa.load(p)[0] for p in paths]
        self.sample_pos = [-1 for _ in self.samples]

    def set_bpm(self, bpm = 120):
        self.bpm = bpm
        ticks_per_beat = 4
        self.tick_length = 60000 / (self.bpm * ticks_per_beat)