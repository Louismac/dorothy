import numpy as np

def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

def apply_fade(signal, sr=22050, fade_duration=0.05):
    fade_samples = int(sr * fade_duration)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    return signal


def sine_step(duration=0.5, midi_notes = range(36, 84)):
    sr = 22050
    samples_per_note = int(sr * duration)
    output = np.array([])
    for midi_note in midi_notes:
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, samples_per_note, False)
        sine_wave = apply_fade(0.5 * np.sin(2 * np.pi * freq * t))
        output = np.concatenate((output, sine_wave))
    output = np.clip(output, -1.0, 1.0, dtype=np.float32)
    return output