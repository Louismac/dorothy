# Dorothy 🎵✨

**Create audio-reactive visuals in Python with just a few lines of code**

Dorothy is a creative computing library that makes it incredibly easy to build interactive visual art that responds to music, beats, and audio in real-time. Think Processing meets Python, with superpowers for audio visualisation.

<img src="images/output2.gif" alt="drawing" width="200"/><img src="images/output3.gif" alt="drawing" width="200"/><img src="images/output4.gif" alt="drawing" width="200"/>

Dorothy in action

---

## ✨ Why Dorothy?

- 🎵 **Audio-First**: Built specifically for music visualisation with FFT, beat detection, and amplitude analysis
- 🎨 **Artist-Friendly**: Processing/p5.js-style API that feels natural for creative coders
- ⚡ **Real-Time**: Smooth visuals with efficient OpenGL rendering
- 🤖 **AI-Ready**: Seamless integration with RAVE, MAGNet, and other ML audio models
- 🐍 **Pure Python**: Use NumPy, PyTorch, TensorFlow, or any Python library alongside Dorothy

---

## 🚀 Quick Start

### Installation
```bash
pip install dorothy-cci
```

### Your First Audio-Reactive Visual
```python
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        # Load your favorite song
        dot.music.start_file_stream("your_song.wav")
        dot.music.play()
    
    def draw(self):
        dot.background((0, 0, 0))  # Black background
        
        # Circle that pulses with the music
        size = 50 + dot.music.amplitude * 200
        dot.fill((255, 100, 150))  # Pink
        dot.circle((dot.width//2, dot.height//2), size)

MySketch()
```

**That's it!** You now have a pink circle that pulses to your music. Press `q` to quit.

---

## 🎯 What Can You Build?

<details>
<summary><strong>🎵 Audio Visualisers</strong> - Spectrum analysers, waveform displays, beat-reactive patterns</summary>

```python
def draw(self):
    dot.background((0, 0, 0))
    
    # Draw FFT bars
    for i, freq in enumerate(dot.music.fft_vals()[:50]):
        height = freq * 300
        x = i * (dot.width / 50)
        dot.fill((100, 200, 255))
        dot.rectangle((x, dot.height - height), (x + 10, dot.height))
```
</details>

<details>
<summary><strong>🎨 Interactive Art</strong> - Mouse-controlled visuals, webcam integration, generative patterns</summary>

```python
def draw(self):
    # Mouse-controlled brush with audio-reactive size
    brush_size = 10 + dot.music.amplitude() * 50
    dot.fill((255, dot.mouse_x % 255, dot.mouse_y % 255))
    dot.circle((dot.mouse_x, dot.mouse_y), brush_size)
```
</details>

<details>
<summary><strong>🤖 AI-Powered Visuals</strong> - RAVE model integration, neural audio synthesis, ML-driven art</summary>

```python
# Generate audio with AI and visualise it
rave_id = dot.music.start_rave_stream("vintage.ts")
dot.music.play()

def draw(self):
    # Visualise AI-generated audio spectrum
    for i, val in enumerate(dot.music.fft_vals):
        # Your visualization code here
```
</details>

---

## 📚 Learning Path

### [All Examples](examples/)

### 🌟 **Level 1: Your First Steps**
1. [🎵 Pulse Rectangle](examples/amplitude.py) - Circle that grows with music
2. [🌈 Color Beats](examples/beats.py) - Colors that change on beats
3. [📊 Simple Spectrum](examples/fft.py) - Your first FFT visualiser

### 🔥 **Level 2: Getting Creative**
4. [🎭 Mouse Magic](examples/mouse.py) - Interactive drawing with audio
5. [📹 Webcam Reaktor](examples/webcam.py) - Video effects with music
6. [✨ Body Tracking](examples/hands.py) - Hand tracking from tensorflow

### 🚀 **Level 3: Advanced Wizardry**
7. [🤖 AI Audio Generation](examples/rave.py) - RAVE and MAGNet integration
8. [🎪 Live Coding](examples/livecode/) - Update visuals without restarting
9. [🎨 Complex Compositions](examples/contours.py) - Multi-layer masterpieces

---

## 🎛️ Core Features

### Audio Analysis
```python
dot.music.amplitude()      # Current volume level (0-1)
dot.music.fft_vals()       # Frequency spectrum array
dot.music.is_beat()      # True if beat detected this frame
```

### Drawing Tools
```python
dot.fill((r, g, b))                  # Set fill color
dot.stroke((r, g, b))                # Set outline color
dot.circle((x, y), radius)           # Draw circles
dot.rectangle((x1, y1), (x2, y2))    # Draw rectangles
dot.line((x1, y1), (x2, y2))         # Draw lines

```

### Interaction
```python
dot.mouse_x, dot.mouse_y             # Mouse position
dot.width, dot.height                # Canvas dimensions
dot.millis, dot.frames               # Time and frame count
```

---

## 🎵 Audio Sources

### 🎧 Play Audio Files
```python
dot.music.start_file_stream("song.wav")
dot.music.play()
```

### 🎤 Live Audio Input
```python
# Use your microphone or system audio
dot.music.start_device_stream(device_id)
```

### 🤖 AI Audio Generation
```python
# Generate with RAVE models
dot.music.start_rave_stream("model.ts")

# Generate with MAGNet models  
dot.music.start_magnet_stream("model.pth", "source.wav")
```

---

## 💡 Pro Tips

- **Start Simple**: Begin with basic shapes and gradually add complexity
- **Use Live Coding**: Enable hot-reloading for faster iteration ([see example](examples/livecode/))
- **Layer Effects**: Use `dot.get_layer()` for transparency and complex compositions
- **Optimise for Performance**: Complex drawings may need optimisation
- **Debug Visually**: Use `annotate=True` on shapes to see coordinates

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.10, 3.11, 3.12, 3.13
- Windows, macOS, or Linux
- Audio device (speakers/headphones recommended)

### For Audio Routing (macOS)
We recommend [BlackHole](https://existential.audio/blackhole/) for routing system audio to Dorothy.

### Troubleshooting
<details>
<summary>Common issues and solutions</summary>

**No audio detected**: Check your audio device with `print(sd.query_devices())`

**Window won't close**: Use `Ctrl+C` in terminal or `q` key with window focused

**Installation issues**: Try `pip3 install dorothy-cci` or create a virtual environment

**Performance issues**: Reduce canvas size or simplify drawing operations
</details>

---

## 🎨 Gallery

*Coming Soon: Amazing projects built with Dorothy by the community!*

Want to showcase your Dorothy creation? [Open an issue](https://github.com/Louismac/dorothy/issues) with your project!

---

## 🤝 Community & Support

- 📖 **Documentation**: [Full API Reference](Reference.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Louismac/dorothy/discussions)
- 🐛 **Issues**: [Report Bugs](https://github.com/Louismac/dorothy/issues)
- 📧 **Contact**: [Your contact info]

---

## 🚀 Contributing

Dorothy is open source and we love contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features  
- 📖 Documentation improvements
- 🎨 Example projects
- 💡 Feature ideas

Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

---

## 📄 License

Dorothy is MIT licensed. Create amazing things! 

---

## 🙏 Acknowledgments

Built with love using:
- [OpenCV](https://opencv.org/) for fast graphics
- [sounddevice](https://python-sounddevice.readthedocs.io/) for audio I/O
- [librosa](https://librosa.org/) for audio analysis
- [NumPy](https://numpy.org/) for efficient computing

Inspired by [Processing](https://processing.org/) and [p5.js](https://p5js.org/) - making creative coding accessible to everyone.

---

<div align="center">

**Ready to make some visual music?** 🎵✨

[Get Started](#-quick-start) • [Examples](examples/) • [Documentation](Reference.md) • [Community](https://github.com/Louismac/dorothy/discussions)

*Made with ❤️ for creative coders, digital artists, and music lovers*

</div>