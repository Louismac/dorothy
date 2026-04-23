from dorothy import Dorothy
from dorothy.Audio.effects import LowPassFilter, Reverb, Delay, Chorus

dot = Dorothy()

class MySketch:

    def setup(self):
        # Load a granular synth as the audio source
        gran_idx = dot.music.start_granular_stream(
            "../audio/gospel.wav",
            density=10,
            grain_size=200,
            spread=0.05,
        )
        self.gran = dot.music.audio_outputs[gran_idx]
        self.gran_idx = gran_idx
        self.gran.note_on(440, vel=0.8)

        # Build an effects chain: LPF -> Reverb -> Delay -> Chorus
        self.lpf    = LowPassFilter(cutoff=2000, q=1.2)
        self.reverb = Reverb(room_size=0.6, damping=0.5, wet=0.35)
        self.delay  = Delay(feedback=0.35, wet=0.4)
        self.chorus = Chorus(rate=0.4, depth=0.003, wet=0.3)

        self.gran.add_effect(self.lpf)
        self.gran.add_effect(self.reverb)
        self.gran.add_effect(self.delay)
        self.gran.add_effect(self.chorus)

    def draw(self):
        dot.background(dot.darkblue)

        # Mouse X → filter cutoff (200 Hz – 8000 Hz)
        self.lpf.cutoff = 200 + (dot.mouse_x / dot.width) * 7800

        # Mouse Y → reverb room size (0.1 – 0.95)
        self.reverb.room_size = 0.1 + (1.0 - dot.mouse_y / dot.height) * 0.85

        # Visualise amplitude as a circle
        amp = dot.music.amplitude(self.gran_idx)
        dot.fill(dot.white)
        dot.circle((dot.width / 2, dot.height / 2), amp * 300)

        # Show parameter values
        dot.fill(dot.white)
        dot.text(f"LPF cutoff:  {self.lpf.cutoff:.0f} Hz",   20, 20)
        dot.text(f"Reverb room: {self.reverb.room_size:.2f}", 20, 45)

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
