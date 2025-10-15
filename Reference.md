# Dorothy API Reference

Complete reference for Dorothy - A Creative Computing Python Library with ModernGL backend

## Table of Contents

- [Getting Started](#getting-started)
- [Main Classes](#main-classes)
- [Drawing Functions](#drawing-functions)
- [Transform Functions](#transform-functions)
- [Layer System](#layer-system)
- [Image Functions](#image-functions)
- [Camera Functions](#camera-functions)
- [Properties](#properties)
- [Color Constants](#color-constants)
- [Audio Integration](#audio-integration)
- [Live Coding](#live-coding)

---

## Getting Started

### Installation

```bash
pip install moderngl moderngl-window PyGLM numpy
```

### Basic Template

```python
from dorothy import Dorothy

dot = Dorothy(width=800, height=600, title="My Sketch")

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("Setup runs once")
    
    def draw(self):
        dot.background((30, 30, 40))
        dot.fill((255, 100, 100))
        dot.circle((400, 300), 50)

MySketch()
```

---

## Main Classes

### Dorothy

Main class for creating Dorothy applications.

#### Constructor

```python
Dorothy(width=800, height=600, title="Dorothy")
```

**Parameters:**
- `width` (int): Window width in pixels (default: 800)
- `height` (int): Window height in pixels (default: 600)
- `title` (str): Window title (default: "Dorothy")

**Example:**
```python
dot = Dorothy(width=1920, height=1080, title="My Project")
```

#### Methods

##### start_loop(setup_fn, draw_fn)

Start the main render loop.

**Parameters:**
- `setup_fn` (Callable): Function to run once at startup
- `draw_fn` (Callable): Function to run every frame

**Example:**
```python
def setup():
    dot.background((255, 255, 255))

def draw():
    dot.circle((dot.mouse_x, dot.mouse_y), 20)

dot.start_loop(setup, draw)
```

##### start_livecode_loop(sketch_module)

Start a live coding loop that reloads code on file changes.

**Parameters:**
- `sketch_module`: Python module containing a `MySketch` class

**Example:**
```python
import my_sketch
dot.start_livecode_loop(my_sketch)
```

---

## Drawing Functions

All drawing functions respect the current fill, stroke, and transform state.

### 2D Shapes

#### circle(center, radius, annotate=False)

Draw a circle.

**Parameters:**
- `center` (Tuple[float, float]): (x, y) center position
- `radius` (float): Circle radius
- `annotate` (bool): Draw debug cross at center (default: False)

**Example:**
```python
dot.fill((255, 0, 0))
dot.stroke((0, 0, 0))
dot.set_stroke_weight(2)
dot.circle((400, 300), 100)

# With annotation for debugging
dot.circle((200, 200), 50, annotate=True)
```

#### rectangle(pos1, pos2, annotate=False)

Draw a rectangle.

**Parameters:**
- `pos1` (Tuple[float, float]): (x1, y1) top-left corner
- `pos2` (Tuple[float, float]): (x2, y2) bottom-right corner
- `annotate` (bool): Draw debug marker (default: False)

**Example:**
```python
dot.fill((0, 255, 0))
dot.rectangle((100, 100), (300, 200))
```

#### line(pos1, pos2, annotate=False)

Draw a line.

**Parameters:**
- `pos1` (Tuple[float, float]): (x1, y1) start point
- `pos2` (Tuple[float, float]): (x2, y2) end point
- `annotate` (bool): Draw debug marker (default: False)

**Example:**
```python
dot.stroke((255, 255, 0))
dot.set_stroke_weight(5)
dot.line((0, 0), (800, 600))
```

### 3D Shapes

#### sphere(radius=1.0)

Draw a 3D sphere (requires 3D camera mode).

**Parameters:**
- `radius` (float): Sphere radius (default: 1.0)

**Example:**
```python
dot.camera_3d()
dot.set_camera((0, 0, 5), (0, 0, 0))
dot.fill((255, 100, 100))
dot.sphere(1.0)
```

#### box(width=1.0, height=1.0, depth=1.0)

Draw a 3D box (requires 3D camera mode).

**Parameters:**
- `width` (float): Box width (default: 1.0)
- `height` (float): Box height (default: 1.0)
- `depth` (float): Box depth (default: 1.0)

**Example:**
```python
dot.camera_3d()
dot.fill((100, 100, 255))
dot.box(2.0, 1.0, 1.5)
```

### Style Functions

#### fill(color)

Set fill color for shapes.

**Parameters:**
- `color` (Tuple[int, int, int] or Tuple[int, int, int, int]): RGB or RGBA color (0-255)

**Example:**
```python
dot.fill((255, 0, 0))        # Red
dot.fill((0, 255, 0, 128))   # Green, 50% transparent
```

#### no_fill()

Disable fill for shapes (only stroke will be drawn).

**Example:**
```python
dot.no_fill()
dot.stroke((0, 0, 0))
dot.circle((400, 300), 50)  # Hollow circle
```

#### stroke(color)

Set stroke color for shape outlines.

**Parameters:**
- `color` (Tuple[int, int, int] or Tuple[int, int, int, int]): RGB or RGBA color (0-255)

**Example:**
```python
dot.stroke((0, 0, 255))
dot.set_stroke_weight(3)
```

#### no_stroke()

Disable stroke for shapes (only fill will be drawn).

**Example:**
```python
dot.fill((255, 0, 0))
dot.no_stroke()
dot.circle((400, 300), 50)  # Filled circle, no outline
```

#### set_stroke_weight(weight)

Set stroke line width.

**Parameters:**
- `weight` (float): Line width in pixels

**Example:**
```python
dot.set_stroke_weight(5)
dot.line((0, 0), (100, 100))
```

#### background(color)

Clear the screen with a color.

**Parameters:**
- `color` (Tuple[int, int, int]): RGB color (0-255)

**Example:**
```python
dot.background((30, 30, 40))  # Dark gray
```

---

## Transform Functions

Transforms apply to all subsequent drawing until reset or popped.

### push_matrix()

Save the current transformation state.

**Example:**
```python
dot.push_matrix()
dot.translate(400, 300)
dot.rotate(0.5)
dot.circle((0, 0), 50)  # Rotated circle at (400, 300)
dot.pop_matrix()        # Restore original state
```

### pop_matrix()

Restore the previously saved transformation state.

### translate(x, y, z=0)

Translate (move) the coordinate system.

**Parameters:**
- `x` (float): X-axis translation
- `y` (float): Y-axis translation
- `z` (float): Z-axis translation (default: 0)

**Example:**
```python
dot.translate(400, 300)
dot.circle((0, 0), 50)  # Circle at (400, 300)
```

### rotate(angle, x=0, y=0, z=1)

Rotate the coordinate system.

**Parameters:**
- `angle` (float): Rotation angle in **radians**
- `x` (float): X component of rotation axis (default: 0)
- `y` (float): Y component of rotation axis (default: 0)
- `z` (float): Z component of rotation axis (default: 1 for 2D rotation)

**Example:**
```python
import math

# 2D rotation (around Z axis)
dot.translate(400, 300)
dot.rotate(math.pi / 4)  # 45 degrees
dot.rectangle((-50, -50), (50, 50))

# 3D rotation (around Y axis)
dot.rotate(math.pi / 6, x=0, y=1, z=0)
```

### scale(s, y=None, z=None)

Scale the coordinate system.

**Parameters:**
- `s` (float): X-axis scale factor (or uniform scale if y and z are None)
- `y` (float): Y-axis scale factor (default: same as s)
- `z` (float): Z-axis scale factor (default: same as s)

**Example:**
```python
# Uniform scaling
dot.scale(2.0)
dot.circle((0, 0), 50)  # Circle with radius 100

# Non-uniform scaling
dot.scale(2.0, 0.5)
dot.circle((0, 0), 50)  # Ellipse
```

### reset_transforms()

Reset all transformations to identity.

**Example:**
```python
dot.translate(100, 100)
dot.rotate(1.0)
dot.reset_transforms()  # Back to original state
```

### Transform Pattern for Scaling Around Center

```python
# To scale around a specific point (e.g., center of image):
center_x, center_y = 400, 300

dot.push_matrix()
dot.translate(center_x, center_y, 0)     # Move to center
dot.scale(2.0)                            # Scale from center
dot.translate(-width/2, -height/2, 0)    # Offset for drawing
dot.paste(image, (0, 0))                 # Image scales from center
dot.pop_matrix()
```

---

## Layer System

Layers are offscreen framebuffers for advanced compositing and effects.

### get_layer()

Create a new layer.

**Returns:**
- `int`: Layer ID

**Example:**
```python
layer = dot.get_layer()
```

### begin_layer(layer_id)

Start rendering to a layer.

**Parameters:**
- `layer_id` (int): Layer ID from `get_layer()`

**Example:**
```python
layer = dot.get_layer()
dot.begin_layer(layer)
dot.circle((400, 300), 100)
dot.end_layer()
```

### end_layer()

Stop rendering to layer, return to screen.

**Example:**
```python
dot.end_layer()
```

### draw_layer(layer_id, alpha=1.0)

Draw a layer to the screen with transparency.

**Parameters:**
- `layer_id` (int): Layer ID to draw
- `alpha` (float): Transparency (0.0 = invisible, 1.0 = opaque)

**Example:**
```python
dot.draw_layer(layer, alpha=0.5)  # Draw layer at 50% opacity
```

### release_layer(layer_id)

Free a layer's resources when done.

**Parameters:**
- `layer_id` (int): Layer ID to release

**Example:**
```python
dot.release_layer(layer)
```

### Layer Examples

#### Trail Effect

```python
class TrailSketch:
    def setup(self):
        self.trail_layer = dot.get_layer()
    
    def draw(self):
        dot.background((0, 0, 0))
        
        # Fade previous trails
        dot.begin_layer(self.trail_layer)
        dot.fill((0, 0, 0))  # Black with some transparency
        dot.rectangle((0, 0), (800, 600))
        
        # Draw new content
        dot.fill((255, 100, 100))
        dot.circle((dot.mouse_x, dot.mouse_y), 20)
        dot.end_layer()
        
        # Composite to screen
        dot.draw_layer(self.trail_layer)
```

#### Multiple Layers

```python
def setup(self):
    self.bg_layer = dot.get_layer()
    self.fg_layer = dot.get_layer()
    
    # Draw static background once
    dot.begin_layer(self.bg_layer)
    for i in range(100):
        dot.fill((100, 100, 200))
        dot.circle((random() * 800, random() * 600), 5)
    dot.end_layer()

def draw(self):
    dot.background((0, 0, 0))
    
    # Draw background layer
    dot.draw_layer(self.bg_layer, alpha=0.3)
    
    # Draw foreground to layer
    dot.begin_layer(self.fg_layer)
    dot.background((0, 0, 0))
    dot.fill((255, 255, 0))
    dot.circle((dot.mouse_x, dot.mouse_y), 50)
    dot.end_layer()
    
    # Composite foreground
    dot.draw_layer(self.fg_layer, alpha=0.8)
```

---

## Image Functions

### paste(image, position, size=None, alpha=1.0)

Paste a NumPy array (image) onto the canvas.

**Parameters:**
- `image` (np.ndarray): Image array. Supports:
  - `(H, W, 3)` - RGB
  - `(H, W, 4)` - RGBA
  - `(H, W)` - Grayscale
  - Values: uint8 (0-255) or float (0.0-1.0)
- `position` (Tuple[int, int]): (x, y) top-left corner
- `size` (Tuple[int, int], optional): (width, height) to resize. None = original size
- `alpha` (float): Overall transparency (0.0-1.0)

**Respects transforms!** Use push/pop/translate/scale for positioning and effects.

**Example:**
```python
import cv2

# Load image
img = cv2.imread('photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Paste at position
dot.paste(img, (100, 100))

# Paste with resize
dot.paste(img, (200, 200), size=(400, 300))

# Paste with transparency
dot.paste(img, (0, 0), alpha=0.5)

# Paste with transform (scale from center)
dot.push_matrix()
dot.translate(400, 300)
dot.scale(2.0)
dot.translate(-img.shape[1]//2, -img.shape[0]//2)
dot.paste(img, (0, 0))
dot.pop_matrix()
```

### Working with OpenCV

```python
import cv2

# Load image
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load PNG with alpha
logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)

# Resize
img_small = cv2.resize(img, (320, 240))

# Webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
dot.paste(frame, (0, 0))
```

---

## Camera Functions

### camera_2d()

Switch to 2D orthographic camera (default mode).

**Example:**
```python
dot.camera_2d()
dot.circle((400, 300), 50)  # Standard 2D drawing
```

### camera_3d()

Switch to 3D perspective camera.

**Example:**
```python
dot.camera_3d()
dot.set_camera((0, 0, 5), (0, 0, 0))
dot.sphere(1.0)
```

### set_camera(eye, target=(0, 0, 0))

Set 3D camera position and look-at target.

**Parameters:**
- `eye` (Tuple[float, float, float]): Camera position (x, y, z)
- `target` (Tuple[float, float, float]): Look-at point (default: origin)

**Example:**
```python
# Camera at (5, 3, 5) looking at origin
dot.set_camera((5, 3, 5), (0, 0, 0))

# Orbit camera
angle = dot.frames * 0.01
x = 5 * math.cos(angle)
z = 5 * math.sin(angle)
dot.set_camera((x, 2, z), (0, 0, 0))
```

---

## Properties

Properties are read-only and updated automatically.

### mouse_x

Current mouse X coordinate.

**Type:** `int`

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 20)
```

### mouse_y

Current mouse Y coordinate.

**Type:** `int`


### width

Window width in pixels.

**Type:** `int`

**Example:**
```python
center_x = dot.width // 2
dot.circle((center_x, 300), 50)
```

### height

Window height in pixels.

**Type:** `int`

### frames

Number of frames rendered since start.

**Type:** `int`

**Example:**
```python
# Animate based on frame count
x = 400 + 200 * math.sin(dot.frames * 0.05)
dot.circle((x, 300), 50)
```

### millis

Time in milliseconds since start.

**Type:** `float`

**Example:**
```python
# Animate based on time
t = dot.millis / 1000  # Convert to seconds
x = 400 + 200 * math.sin(t * 2)
dot.circle((x, 300), 50)
```

---

## Interaction Callbacks

### on_mouse_pressed(x,y,button)

**Example:**
```python
self.color = dot.red
def mouse_pressed(x,y,b):
    if self.color == dot.red:
        self.color = dot.blue
    else:
        self.color = dot.red
dot.on_mouse_press = mouse_pressed
```

### on_mouse_released(x,y,button)

### on_scroll(x_offset,x_offset)

### on_mouse_drag(x,y,dx,dy)

### on_key_press(key, action, modifiers)

#### Available Constants:
##### Action Types:

* dot.keys.ACTION_PRESS
* dot.keys.ACTION_RELEASE

##### Common Keys:

* dot.keys.SPACE
* dot.keys.ENTER
* dot.keys.ESCAPE
* dot.keys.TAB
* dot.keys.BACKSPACE
* Letters: dot.keys.A through dot.keys.Z
* Numbers: dot.keys.NUMBER_0 through dot.keys.NUMBER_9
* Arrows: dot.keys.UP, dot.keys.DOWN, dot.keys.LEFT, dot.keys.RIGHT

##### Modifiers:
 
* dot.modifiers.shift
* dot.modifiers.ctrl
* dot.modifiers.alt

**Example:**
```python
self.color = dot.red

def key_press(key, action, modifiers):
    if action == dot.keys.ACTION_PRESS:
        if key == dot.keys.SPACE:
            print("SPACE key was pressed")
        if key == dot.keys.Z and modifiers.shift:
            print("Shift + Z was pressed")

        if key == dot.keys.Z and modifiers.ctrl:
            print("ctrl + Z was pressed")
    elif action == dot.keys.ACTION_RELEASE:
        if key == dot.keys.SPACE:
            print("SPACE key was released")

    if self.color == dot.red:
        self.color = dot.blue
    else:
        self.color = dot.red

dot.on_key_press = key_press
```



---

## Color Constants

Predefined colors for convenience (all css colours available).

**Example:**
```python
dot.fill(dot.red)
dot.circle((400, 300), 50)
```

---

## Audio Integration

Dorothy integrates with a comprehensive audio system for playback, analysis, and generation. The audio system runs on separate threads to avoid interfering with rendering.

### Setup

```python
from dorothy import Dorothy
from dorothy.audio import Audio  # Or your audio module path

dot = Dorothy()
dot.music = Audio()
```

### Audio Sources

Dorothy supports multiple audio sources that can run simultaneously:

#### File Playback

```python
# Play an audio file
file_id = dot.music.start_file_stream(
    "song.wav",
    fft_size=512,        # FFT window size for analysis
    buffer_size=1024,    # Audio buffer size (larger = smoother, more latency)
    sr=44100,            # Sample rate
    output_device=None,  # None = default output, or device ID
    analyse=True         # Enable FFT and amplitude analysis
)

dot.music.play(file_id)
```

#### Device Input (Microphone/System Audio)

```python
# Capture from microphone or system audio
device_id = dot.music.start_device_stream(
    device=2,            # Device ID (see sd.query_devices())
    fft_size=1024,
    buffer_size=2048,
    sr=44100,
    analyse=True
)

dot.music.play(device_id)
```

#### Custom Audio Generation (DSP)

```python
# Generate audio with a callback function
phase = 0

def audio_callback(size):
    global phase
    frequency = 440  # A4 note
    sr = 44100
    delta = 2 * np.pi * frequency / sr
    x = delta * np.arange(size)
    audio = 0.3 * np.sin(phase + x)
    phase += delta * size
    return audio

dsp_id = dot.music.start_dsp_stream(
    audio_callback,
    fft_size=512,
    buffer_size=1024,
    sr=44100,
    output_device=None,
    analyse=True
)
```

#### Sample Playback

```python
# Play pre-loaded samples
import numpy as np

# Generate or load samples
samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)

sample_id = dot.music.start_sample_stream(
    samples,
    fft_size=1024,
    buffer_size=1024,
    sr=44100,
    output_device=None,
    analyse=True
)
```

#### RAVE Model Generation

```python
# Generate audio with RAVE neural vocoder
rave_id = dot.music.start_rave_stream(
    model_path="vintage.ts",
    fft_size=1024,
    buffer_size=2048,
    sr=44100,
    latent_dim=16,       # Must match model
    output_device=None
)

# Control latent space
z = torch.randn(1, 16, 1)
dot.music.audio_outputs[rave_id].current_latent = z

# Add bias to latent
dot.music.audio_outputs[rave_id].z_bias = torch.randn(1, 16, 1) * 0.1
```

#### MAGNet Model Generation

```python
# Generate audio with MAGNet spectral model
magnet_id = dot.music.start_magnet_stream(
    model_path="model.pth",
    dataset_path="audio_samples.wav",
    buffer_size=2048,
    sr=44100,
    output_device=None
)
```

### Analysis Properties

All audio sources (when `analyse=True`) provide real-time analysis:

#### amplitude(output=0, smooth=1)

Get current audio amplitude (volume).

**Parameters:**
- `output` (int): Audio output index (default: 0)
- `smooth` (int): Smoothing factor (default: 1)

**Returns:**
- `float`: RMS amplitude (0.0-1.0+)

**Example:**
```python
amp = dot.music.amplitude()
radius = 50 + amp * 200
dot.circle((400, 300), radius)
```

#### fft(output=0)

Get current FFT frequency spectrum.

**Parameters:**
- `output` (int): Audio output index (default: 0)

**Returns:**
- `np.ndarray`: FFT magnitudes, length = (fft_size // 2) + 1

**Example:**
```python
fft = dot.music.fft()
for i, magnitude in enumerate(fft[::4]):  # Every 4th bin
    x = i * 20
    height = magnitude * 300
    dot.rectangle((x, 600), (x + 18, 600 - height))
```

#### is_beat(output=0)

Detect if a beat occurred since last call.

**Parameters:**
- `output` (int): Audio output index (default: 0)

**Returns:**
- `bool`: True if beat detected

**Example:**
```python
if dot.music.is_beat():
    dot.fill((255, 0, 0))
    dot.circle((400, 300), 200)
```

### Playback Control

#### play(output=0)

Start or resume audio playback.

```python
dot.music.play()        # Play first output
dot.music.play(1)       # Play second output
```

#### stop(output=0)

Stop audio playback completely.

```python
dot.music.stop()
```

#### pause(output=0)

Pause audio playback (can be resumed).

```python
dot.music.pause()
```

#### resume(output=0)

Resume paused audio.

```python
dot.music.resume()
```

### Multiple Audio Outputs

You can have multiple audio sources running simultaneously:

```python
class MultiAudioSketch:
    def setup(self):
        # Background music
        self.music_id = dot.music.start_file_stream("background.wav")
        
        # Microphone input for visualization
        self.mic_id = dot.music.start_device_stream(device=0)
        
        # Synth for interaction
        def synth(size):
            freq = 440 + dot.mouse_x
            return 0.1 * np.sin(2 * np.pi * freq * np.arange(size) / 44100)
        self.synth_id = dot.music.start_dsp_stream(synth)
        
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
```

### Advanced Features

#### Timbre Transfer with RAVE

Route one audio source through a RAVE model:

```python
# Start RAVE generator
rave_id = dot.music.start_rave_stream("vintage.ts")

# Start microphone input
mic_id = dot.music.start_device_stream(device=0)

# Route mic through RAVE
dot.music.update_rave_from_stream(mic_id)

# Now RAVE will encode mic input and generate audio
```

#### Custom Callbacks

Access raw audio buffers:

```python
def on_new_frame(buffer):
    # Called when new audio buffer is available
    print(f"New audio: {buffer.shape}, max: {np.max(np.abs(buffer))}")

file_id = dot.music.start_file_stream("song.wav")
dot.music.audio_outputs[file_id].on_new_frame = on_new_frame
```

#### Gain Control

Adjust output volume per source:

```python
file_id = dot.music.start_file_stream("song.wav")

# Set gain (volume multiplier)
dot.music.audio_outputs[file_id].gain = 0.5  # 50% volume

# Mute
dot.music.audio_outputs[file_id].gain = 0.0
```

### Query Available Devices

```python
import sounddevice as sd

# List all audio devices
print(sd.query_devices())

# Get default devices
print(sd.default.device)  # [input_device, output_device]
```

### Audio Performance Tips

1. **Buffer Size**: Larger buffers = smoother audio but more latency
   - For live input: 512-1024
   - For playback: 1024-2048
   - If glitching: try 4096

2. **FFT Size**: Balance between frequency resolution and time resolution
   - Music: 1024-2048
   - Speech: 512-1024
   - Real-time: 512

3. **Cache Audio Data**: Only read once per frame
```python
def draw(self):
    # GOOD: Read once
    fft = dot.music.fft()
    amp = dot.music.amplitude()
    
    # Use cached values
    for i, val in enumerate(fft):
        pass
    
    # BAD: Reading multiple times
    # for i in range(len(dot.music.fft())):
    #     val = dot.music.fft()[i]  # Don't do this!
```

4. **Reduce Visual Complexity**: If audio glitches, simplify drawing
```python
# Subsample FFT for fewer bars
fft = dot.music.fft()[::4]  # Every 4th value
```

### Complete Audio Examples

#### Example 1: FFT Visualizer

```python
class FFTVisualizer:
    def setup(self):
        self.file_id = dot.music.start_file_stream(
            "song.wav",
            fft_size=2048,
            buffer_size=2048
        )
        dot.music.play()
    
    def draw(self):
        dot.background((10, 10, 15))
        
        # Get FFT data
        fft = dot.music.fft()
        
        # Draw frequency bars
        bar_width = dot.width / len(fft[::4])
        for i, magnitude in enumerate(fft[::4]):
            x = i * bar_width
            height = magnitude * 500
            
            # Color based on frequency
            hue = i / len(fft[::4])
            r = int(255 * (1 - hue))
            g = int(255 * hue)
            
            dot.fill((r, g, 200))
            dot.no_stroke()
            dot.rectangle((x, 600), (x + bar_width - 2, 600 - height))
```

#### Example 2: Beat Detection

```python
class BeatReactive:
    def setup(self):
        dot.music.start_file_stream("song.wav")
        dot.music.play()
        self.beat_time = 0
        self.beat_intensity = 0
    
    def draw(self):
        dot.background((20, 20, 30))
        
        # Detect beats
        if dot.music.is_beat():
            self.beat_time = dot.frames
            self.beat_intensity = 1.0
        
        # Fade beat intensity
        self.beat_intensity *= 0.95
        
        # Draw pulsing circle on beat
        radius = 100 + self.beat_intensity * 200
        alpha = int(self.beat_intensity * 255)
        dot.fill((255, 100, 100, alpha))
        dot.circle((400, 300), radius)
        
        # Amplitude meter
        amp = dot.music.amplitude()
        dot.fill((100, 255, 100))
        dot.rectangle((50, 550), (50 + amp * 700, 570))
```

#### Example 3: Microphone Reactive

```python
class MicReactive:
    def setup(self):
        # Find microphone device
        import sounddevice as sd
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if 'microphone' in device['name'].lower():
                print(f"Using device {i}: {device['name']}")
                self.mic_id = dot.music.start_device_stream(
                    device=i,
                    fft_size=1024,
                    buffer_size=1024
                )
                break
    
    def draw(self):
        dot.background((15, 15, 20))
        
        # Get microphone FFT
        fft = dot.music.fft(self.mic_id)
        amp = dot.music.amplitude(self.mic_id)
        
        # Radial frequency visualization
        for i, magnitude in enumerate(fft[::2]):
            angle = (i / len(fft[::2])) * 2 * np.pi
            length = 100 + magnitude * 200
            
            x1 = 400 + 100 * np.cos(angle)
            y1 = 300 + 100 * np.sin(angle)
            x2 = 400 + length * np.cos(angle)
            y2 = 300 + length * np.sin(angle)
            
            dot.stroke((100, 200, 255))
            dot.set_stroke_weight(2)
            dot.line((x1, y1), (x2, y2))
        
        # Center amplitude circle
        dot.fill((255, 100, 100))
        dot.no_stroke()
        dot.circle((400, 300), 30 + amp * 70)
```

#### Example 4: RAVE Interactive

```python
class RAVEInteractive:
    def setup(self):
        self.rave_id = dot.music.start_rave_stream(
            "vintage.ts",
            latent_dim=16
        )
        self.z = torch.randn(1, 16, 1)
        dot.music.play(self.rave_id)
    
    def draw(self):
        dot.background((25, 25, 35))
        
        # Update latent based on mouse
        mouse_x_norm = dot.mouse_x / dot.width
        mouse_y_norm = dot.mouse_y / dot.height
        
        # Smoothly interpolate latent
        target_z = torch.randn(1, 16, 1) * mouse_x_norm
        self.z = 0.95 * self.z + 0.05 * target_z
        
        # Update RAVE
        dot.music.audio_outputs[self.rave_id].current_latent = self.z
        
        # Visualize latent dimensions
        for i in range(16):
            val = float(self.z[0, i, 0])
            x = 50 + i * 45
            height = val * 100
            
            color = (100, 150, 255) if val > 0 else (255, 100, 100)
            dot.fill(color)
            dot.rectangle((x, 300), (x + 40, 300 - height))
        
        # Visualize output
        fft = dot.music.fft(self.rave_id)
        for i, val in enumerate(fft[::8]):
            x = i * 30
            h = val * 200
            dot.fill((255, 200, 100))
            dot.rectangle((x, 600), (x + 28, 600 - h))
```

### Troubleshooting Audio

#### Audio Glitches/Crackling

**Cause**: Render loop blocking audio thread  
**Solution**:
- Increase buffer size: `buffer_size=4096`
- Simplify draw() function
- Reduce FFT size
- Enable VSync (already enabled in ModernGL version)

#### No Audio Output

**Check**:
```python
import sounddevice as sd
print(sd.query_devices())  # List devices
print(sd.default.device)   # Check default
```

**Set device explicitly**:
```python
dot.music.start_file_stream("song.wav", output_device=2)
```

#### FFT Values All Zero

**Causes**:
- `analyse=False` was set
- Buffer size too small
- No audio playing yet

**Solution**:
```python
dot.music.start_file_stream("song.wav", analyse=True, buffer_size=2048)
dot.music.play()
```

#### Beat Detection Not Working

**Requirements**:
- Only works with `start_file_stream()` or `start_sample_stream()`
- Needs actual audio file, not live input
- Beat tracking runs during stream initialization

---

## Live Coding

### Setup

Create two files in the same directory, and make sure you are in this directory when you run the code:

**main.py** (run this):
```python
from dorothy import Dorothy
import my_sketch

dot = Dorothy()
dot.start_livecode_loop(my_sketch)
```

**my_sketch.py** (edit this while running):
```python
class MySketch:
    def setup(self):
        print("Setup - runs on reload")
        self.color = (255, 0, 0)
    
    def draw(self):
        dot.background((30, 30, 40))
        dot.fill(self.color)
        dot.circle((dot.mouse_x, dot.mouse_y), 50)
    
    def run_once(self):
        # Optional: runs once when code changes
        print("Code updated!")
        self.color = (random.randint(0, 255), 
                     random.randint(0, 255), 
                     random.randint(0, 255))
```

Now edit `my_sketch.py` and save - changes appear instantly!

### run_once() Method

Special method that runs only once when the code changes:

```python
def run_once(self):
    # Reset state when code updates
    self.particles = []
    self.angle = 0
    print("Code reloaded, state reset!")
```

---

## Complete Examples

### Example 1: Basic Sketch

```python
from dorothy import Dorothy
import math

dot = Dorothy(800, 600, "Basic Sketch")

class MySketch:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        print("Setup complete!")
    
    def draw(self):
        dot.background((30, 30, 40))
        
        # Rotating circle
        x = 400 + 200 * math.cos(self.angle)
        y = 300 + 200 * math.sin(self.angle)
        
        dot.fill((255, 100, 100))
        dot.no_stroke()
        dot.circle((x, y), 30)
        
        self.angle += 0.05

MySketch()
```

### Example 2: Audio Reactive

```python
from dorothy import Dorothy

dot = Dorothy()
dot.music = YourMusicPlayer()

class AudioSketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.music.start_file_stream("song.wav")
        dot.music.play()
    
    def draw(self):
        dot.background((20, 20, 30))
        
        # FFT bars
        fft = dot.music.fft_vals
        for i, val in enumerate(fft[::4]):  # Every 4th value
            x = i * 20
            h = val * 400
            
            # Color based on frequency
            r = int(i / len(fft) * 255)
            dot.fill((r, 100, 255 - r))
            dot.no_stroke()
            dot.rectangle((x, 600), (x + 18, 600 - h))
        
        # Center circle pulses with amplitude
        amp = dot.music.amplitude
        dot.fill((255, 255, 255))
        dot.circle((400, 300), 50 + amp * 200)

AudioSketch()
```

### Example 3: Webcam with Effects

```python
import cv2
from dorothy import Dorothy

dot = Dorothy()

class WebcamSketch:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.layer = None
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.music.start_device_stream(2)
        dot.music.play()
        self.layer = dot.get_layer()
    
    def draw(self):
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Process frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop center
        h, w = frame.shape[:2]
        crop = frame[h//4:h*3//4, w//4:w*3//4]
        
        # Clear screen
        dot.background((0, 0, 0))
        
        # Draw to layer with audio reactive scale
        dot.begin_layer(self.layer)
        dot.background((0, 0, 0))
        
        dot.push_matrix()
        # Scale from center based on amplitude
        factor = 1.0 + dot.music.amplitude * 2.0
        dot.translate(dot.width//2, dot.height//2)
        dot.scale(factor)
        dot.translate(-crop.shape[1]//2, -crop.shape[0]//2)
        dot.paste(crop, (0, 0))
        dot.pop_matrix()
        
        dot.end_layer()
        
        # Draw layer to screen
        dot.draw_layer(self.layer)

WebcamSketch()
```

### Example 4: 3D Scene

```python
from dorothy import Dorothy
import math

dot = Dorothy()

class Scene3D:
    def __init__(self):
        self.angle = 0
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.camera_3d()
    
    def draw(self):
        dot.background((20, 20, 30))
        
        # Orbit camera
        x = 5 * math.cos(self.angle)
        z = 5 * math.sin(self.angle)
        dot.set_camera((x, 2, z), (0, 0, 0))
        
        # Draw multiple objects
        for i in range(-2, 3):
            for j in range(-2, 3):
                dot.push_matrix()
                dot.translate(i * 2, 0, j * 2)
                
                # Alternate between spheres and boxes
                if (i + j) % 2 == 0:
                    dot.fill((255, 100, 100))
                    dot.sphere(0.4)
                else:
                    dot.fill((100, 100, 255))
                    dot.box(0.6, 0.6, 0.6)
                
                dot.pop_matrix()
        
        self.angle += 0.01

Scene3D()
```

---

## Tips & Best Practices

### Performance

1. **Minimize state changes** - batch drawing with same colors
2. **Use layers for static content** - draw once, reuse many times
3. **Cache audio data** - read `fft_vals` once per frame
4. **Reduce geometry complexity** - subsample data when possible

### Transforms

1. **Always use push/pop** - wrap transforms to avoid side effects
2. **Order matters** - translate → rotate → scale is typical
3. **Scale from center** - translate to center, scale, translate back

### Debugging

1. **Use annotate=True** - shows shape coordinates
2. **Print frame count** - `if dot.frames % 60 == 0: print(...)`
3. **Check mouse position** - `print(dot.mouse_x, dot.mouse_y)`

### Code Organization

1. **Use class for state** - store variables in `self`
2. **Setup once** - expensive operations in `setup()`
3. **Keep draw() fast** - called 60 times per second

---

## Troubleshooting

### Audio Glitches

- Increase buffer size: `blocksize=4096`
- Simplify draw() function
- Cache audio data once per frame

### Shapes Not Visible

- Check camera mode: `dot.camera_2d()` for 2D
- Check fill/stroke: ensure colors are set
- Check coordinates: are they within window bounds?

### Transforms Not Working

- Use `push_matrix()` / `pop_matrix()`
- Set transforms BEFORE drawing
- Remember transforms accumulate

### Images Upside Down

- Dorothy handles this automatically
- If issues persist, flip with: `img = np.flipud(img)`

### Mouse Not Working

- Mouse position updates via polling every frame
- No action needed - `dot.mouse_x` and `dot.mouse_y` update automatically

---

## Version History

### ModernGL Refactor (Current)

- GPU-accelerated rendering with ModernGL
- Native 3D support
- 10-100x performance improvement
- Full backward compatibility with original API
- Transform-aware image pasting
- Layer system with alpha blending

---

## Credits

Original Dorothy by Louis McCallum  
ModernGL refactor maintains API compatibility while adding GPU acceleration and 3D support.