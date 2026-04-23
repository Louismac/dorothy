"""Run multiple independent audio streams simultaneously.

Use dot.music.list_devices() to find your microphone device name/index.
"""
from dorothy import Dorothy
import numpy as np

dot = Dorothy(800, 600)

class MySketch:

    def setup(self):
        self.music_id = dot.music.start_file_stream("../audio/gospel.wav")

        # Replace with your device name or index from dot.music.list_devices()
        self.mic_id   = dot.music.start_device_stream("MacBook Pro Microphone")

        self.phase = 0.0
        sr = 44100

        def get_frame(size):
            freq  = dot.mouse_x
            amp   = dot.mouse_y / dot.height
            inc   = 2 * np.pi * freq / sr
            audio = amp * np.sin(self.phase + inc * np.arange(size))
            self.phase += inc * size
            return audio

        self.synth_id = dot.music.start_dsp_stream(get_frame, sr=sr)

    def draw(self):
        dot.background((20, 20, 30))

        # Mic FFT as a bar chart
        for i, val in enumerate(dot.music.fft(self.mic_id)[::4]):
            h = val * 300
            dot.fill((100, 200, 255))
            dot.rectangle((i * 20, dot.height - h), (i * 20 + 18, dot.height))

        # File stream amplitude as a circle
        amp = dot.music.amplitude(self.music_id)
        dot.fill((255, 100, 100))
        dot.circle((dot.width // 2, dot.height // 2), 50 + amp * 150)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
