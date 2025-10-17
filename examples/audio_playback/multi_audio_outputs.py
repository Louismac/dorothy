from dorothy import Dorothy
import numpy as np

dot = Dorothy(800,600)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw) 

    def setup(self):
        # Background music
        self.music_id = dot.music.start_file_stream("../audio/gospel.wav")
        
        # Microphone input for visualization
        self.mic_id = dot.music.start_device_stream(device=1)
        
        self.phase = 0.0 
        sr = 44100

        #Audio Callback function
        def get_frame(size):
            #Get parameters from mouse
            frequency = dot.mouse_x
            amplitude = dot.mouse_y/dot.height
            #Get increments
            phase_increment = 2 * np.pi * frequency / sr 
            x = phase_increment * np.arange(size)
            #Make some sound
            audio = amplitude * np.sin(self.phase + x)
            #update phase
            self.phase += phase_increment * size 
            return audio
        
        self.synth_id = dot.music.start_dsp_stream(get_frame, sr = sr)
        
        dot.music.play(self.music_id)
        dot.music.play(self.mic_id)
        dot.music.play(self.synth_id)
    
    def draw(self):
        dot.background((20, 20, 30))
        
        # Visualize mic input
        mic_fft = dot.music.fft(self.mic_id)
        for i, val in enumerate(mic_fft[::4]):
            x = i * 20
            h = val * 300
            dot.fill((100, 200, 255))
            dot.rectangle((x, 600), (x + 18, 600 - h))
        
        # Music amplitude affects size
        music_amp = dot.music.amplitude(self.music_id)
        dot.fill((255, 100, 100))
        dot.circle((400, 300), 50 + music_amp * 150)
        
MySketch()   
    







