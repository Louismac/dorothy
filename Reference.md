# Dorothy API Reference

Complete reference for Dorothy - A Creative Computing Python Library with ModernGL backend

## Table of Contents

- [Getting Started](#getting-started)
- [Main Classes](#main-classes)
- [Drawing Functions](#drawing-functions)
- [LFO Controls](#oscillators-reference)
- [Transform Functions](#transform-functions)
- [Layer System](#layer-system)
- [Image Functions](#image-functions)
- [Camera Functions](#camera-functions)
- [Video Effects](#video-effects)
- [Custom Shaders](#custom-shaders-reference)
- [Properties](#properties)
- [Color Constants](#color-constants)
- [Audio Integration](#audio-integration)
- [Live Coding](#live-coding)
  

---

## Getting Started

### Installation

```bash
pip install dorothy-cci
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

#### polyline([pts],closed=False)

Draw a multipoint line.

**Parameters:**
- `points` (List[int, int]): points of the line
- `closed` (bool): close the line into a shape (beginning to end)

**Example:**
```python
 # Star shape (concave)
points = []
for i in range(10):
    angle = i * np.pi / 5 - np.pi / 2
    r = 100 if i % 2 == 0 else 50
    x = 400 + r * np.cos(angle)
    y = 300 + r * np.sin(angle)
    points.append((x, y))

dot.stroke(dot.red)
dot.set_stroke_weight(10)
dot.polyline(points, closed = True)  
```

#### polygon([pts])

Draw a filled polygon.

**Parameters:**
- `points` (List[int, int]): points of the polygon

**Example:**
```python
 # Star shape (concave)
points = []
for i in range(10):
    angle = i * np.pi / 5 - np.pi / 2
    r = 100 if i % 2 == 0 else 50
    x = 400 + r * np.cos(angle)
    y = 300 + r * np.sin(angle)
    points.append((x, y))

dot.fill(dot.red)
dot.polygon(points)  
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

## Oscillators Reference

Modulate parameters with cyclical oscillators for animation.

### Creating LFOs

#### `get_lfo(osc, freq, range)`
Create an LFO that automatically updates each frame.
```python
lfo_id = dot.get_lfo(osc='sine', freq=1.0, range=(0, 1))
```

**Parameters:**
- `osc` (str): Oscillator type - `'sine'`, `'saw'`, `'square'`, `'triangle'`
- `freq` (float): Frequency in Hz (cycles per second)
- `range` (tuple): `(min, max)` to map oscillator output to

**Returns:** LFO ID (int) to use with `lfo_value()`

---

### Reading LFO Values

#### `lfo_value(lfo_id)`
Get current value of an LFO.
```python
value = dot.lfo_value(lfo_id)
```

**Returns:** Current value mapped to the LFO's range

---

### Oscillator Types

- **`'sine'`**: Smooth wave oscillation
- **`'saw'`**: Linear ramp up, sharp drop
- **`'square'`**: Alternates between min/max
- **`'triangle'`**: Linear ramp up and down

### Examples

See [Examples](examples/shapes/lfos.py)

### Tips

- **Frequency**: `0.5` = slow (2 seconds per cycle), `2.0` = fast (0.5 seconds per cycle)
- **Ranges**: Can be any values - positions, sizes, colors, angles, etc.
- **Multiple LFOs**: Create as many as needed, each with different frequencies for complex motion
- **Modulation**: Change LFO frequency/range dynamically by modifying `dot.lfos[lfo_id]['freq']`

---

## Transform Functions

Transforms apply to all subsequent drawing within a given block.

### with dot.transform():

Save the current transformation state until the block ends.

**Example:**
```python
with dot.transform():
    dot.translate(400, 300)
    dot.rotate(0.5)
    dot.circle((0, 0), 50)  # Rotated circle at (400, 300)
```


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

with dot.transform():
    dot.translate(center_x, center_y, 0)     # Move to center
    dot.scale(2.0)                            # Scale from center
    dot.translate(-width/2, -height/2, 0)    # Offset for drawing
    dot.paste(image, (0, 0))                 # Image scales from center
```

---

## Layer System

Layers are offscreen framebuffers for advanced compositing and effects. There is always an active layer, and if you are not currently drawing to a specific offscreen layer, you are drawing to the main canvas. 

If you are drawing to an offscreen layer, you will not see the outcomes on the main canvas until you `draw_layer()` 

### get_layer()

Create a new layer.

**Returns:**
- `int`: Layer ID

**Example:**
```python
layer = dot.get_layer()
```

### with dot.layer(layer_id)

Start rendering to a layer in a block, end when block ends

**Parameters:**
- `layer_id` (int): Layer ID from `get_layer()`

**Example:**
```python
layer = dot.get_layer()
with dot.layer(layer): # start drawing to layer
    dot.circle((400, 300), 100)
    dot.rectangle((400, 300), (500,500))
# end drawing to layer
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

#### Scale Layer with Transparency

See [Examples](examples/transforms/scale_layer_with_transparency.py)

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

See [Examples](examples/video_and_images/image.py)

### Working with OpenCV

See [Examples](examples/video_and_images/webcam.py)

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

See [Examples](examples/3d/camera_orbit.py)


---
## Video Effects

Apply visual effects to your canvas with these built-in shader methods.

### Effect Methods

#### `pixelate(pixel_size, accumulate)`
Pixelate the canvas into larger blocks.
```python
dot.pixelate(pixel_size=8.0, accumulate=False)
```

**Parameters:**
- `pixel_size` (float): Size of pixel blocks. Larger = more pixelated (default: 8.0)
- `accumulate` (bool): If True, effect builds up over frames (default: False)

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 50)
dot.pixelate(12.0)
```

---

#### `blur(accumulate)`
Apply blur effect to the canvas.
```python
dot.blur(accumulate=False)
```

**Parameters:**
- `accumulate` (bool): If True, effect builds up over frames (default: False)

**Example:**
```python
dot.circle((200, 200), 100)
dot.blur()
```

---

#### `rgb_split(offset, accumulate)`
Split RGB channels for glitch/chromatic aberration effect.
```python
dot.rgb_split(offset=0.01, accumulate=False)
```

**Parameters:**
- `offset` (float): Distance to split channels, range 0.0-0.1 (default: 0.01)
- `accumulate` (bool): If True, effect builds up over frames (default: False)

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 50)
dot.rgb_split(0.02)
```

---

#### `feedback(zoom, accumulate)`
Create feedback/zoom effect with trails.
```python
dot.feedback(zoom=0.98, accumulate=True)
```

**Parameters:**
- `zoom` (float): Zoom factor per frame. <1.0 zooms out, >1.0 zooms in (default: 0.98)
- `accumulate` (bool): Should be True for feedback effects (default: True)

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 20)
dot.feedback(0.99)  # Slow zoom out with trails
```

---

#### `roll(offset_x, offset_y, accumulate)`
Scroll/shift the canvas with wrapping (like `np.roll`).
```python
dot.roll(offset_x=0.0, offset_y=0.0, accumulate=True)
```

**Parameters:**
- `offset_x` (float): Horizontal shift in pixels. Positive = right (default: 0.0)
- `offset_y` (float): Vertical shift in pixels. Positive = down (default: 0.0)
- `accumulate` (bool): Should be True for rolling effects (default: True)

**Example:**
```python
dot.circle((dot.width//2, dot.height//2), 50)
dot.roll(2.0, 0.0)  # Scroll right continuously
```

---

#### `invert(accumulate)`
Invert all colors on the canvas.
```python
dot.invert(accumulate=False)
```

**Parameters:**
- `accumulate` (bool): If True, effect builds up over frames (default: False)

**Example:**
```python
dot.background(dot.white)
dot.fill(dot.red)
dot.circle((200, 200), 100)
dot.invert()  # White becomes black, red becomes cyan
```

---

#### `tile(grid_x, grid_y, accumulate)`
Tile/repeat the canvas in a grid pattern.
```python
dot.tile(grid_x=2, grid_y=2, accumulate=False)
```

**Parameters:**
- `grid_x` (int): Number of horizontal tiles (default: 2)
- `grid_y` (int): Number of vertical tiles (default: 2)
- `accumulate` (bool): If True, effect builds up over frames (default: False)

**Example:**
```python
dot.circle((100, 100), 50)
dot.tile(4, 4)  # Create 4x4 grid of circles
```

---

#### `cutout(color, threshold, accumulate)`
Make pixels of a specific color transparent (chroma key/green screen).
```python
dot.cutout(color, threshold=0.1, accumulate=True)
```

**Parameters:**
- `color` (tuple or constant): RGB color to cut out, e.g. `(0, 0, 0)` or `dot.green`
- `threshold` (float): Color matching tolerance. 0.0 = exact, 0.5 = loose (default: 0.1)
- `accumulate` (bool): If True, effect builds up over frames (default: True)

**Example:**
```python
dot.background(dot.black)
dot.fill(dot.red)
dot.circle((200, 200), 100)
dot.cutout((0, 0, 0))  # Remove black background
```

---

### Accumulating vs Non-Accumulating

**Accumulating (`accumulate=True`)**: Effect modifies the canvas permanently. Subsequent frames build on the modified version. Use for feedback effects, trails, and persistent transformations and for chaining effects together.

**Non-Accumulating (`accumulate=False`)**: Effect is only a display filter. Canvas content remains unchanged. Use for post-processing like blur, pixelate, color correction.

**Example - Accumulating:**
```python
def draw(self):
    dot.circle((dot.mouse_x, dot.mouse_y), 20)
    dot.feedback(0.99, accumulate=True)  # Creates trails
```

**Example - Non-Accumulating:**
```python
def draw(self):
    dot.background(dot.white)
    dot.circle((200, 200), 50)
    dot.pixelate(8.0, accumulate=False)  # Just a visual filter
```

---

### Combining Effects

Chain multiple effects together:
```python
def draw(self):
    dot.circle((dot.mouse_x, dot.mouse_y), 30)
    
    # Apply multiple effects, remember to accumulate!
    dot.feedback(0.98, accumulate = True)  # Zoom trails
    dot.rgb_split(0.015, accumulate = True))  # Glitch
    dot.pixelate(6.0)  # Retro look
```

---


## Custom Shaders Reference

Apply custom GLSL fragment shaders to create visual effects and post-processing.

## Basic Usage

```python
dot.apply_shader(shader_code, accumulate=True, **uniforms)
```

**Parameters:**
- `shader_code` (str): GLSL fragment shader source code
- `accumulate` (bool): 
  - `True`: Shader modifies persistent canvas, effects build up (feedback effects)
  - `False`: Shader is display-only filter, canvas unchanged (post-processing)
- `**uniforms`: Additional uniforms to pass to shader (e.g., `time=1.5`, `amount=0.1`)

## Shader Template

```glsl
#version 330

uniform sampler2D texture0;  // The canvas texture (always available)
uniform vec2 resolution;      // Canvas size in pixels (optional)

in vec2 v_texcoord;          // Texture coordinates (0-1)
out vec4 fragColor;          // Output color

void main() {
    vec4 color = texture(texture0, v_texcoord);
    // Modify color here
    fragColor = color;
}
```

## Accumulating vs Non-Accumulating

### Accumulating (`accumulate=True`)
Shader output **replaces** the persistent canvas. Effects build up over frames.

**Use for:** Feedback effects, trails, decay, generative art

```python
feedback_shader = '''
#version 330
uniform sampler2D texture0;
uniform float fade;
in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    color.rgb *= fade;  // Fade each frame
    fragColor = color;
}
'''

def draw():
    # Don't call background() - let trails accumulate
    dot.circle((dot.mouse_x, dot.mouse_y), 20)
    dot.apply_shader(feedback_shader, accumulate=True, fade=0.98)
```

### Non-Accumulating (`accumulate=False`)
Shader output is **displayed** but canvas remains unchanged. No feedback.

**Use for:** Blur, color grading, pixelation, distortion

```python
blur_shader = '''
#version 330
uniform sampler2D texture0;
uniform vec2 resolution;
in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    vec2 pixel = 1.0 / resolution;
    vec4 color = vec4(0.0);
    
    for(int x = -2; x <= 2; x++) {
        for(int y = -2; y <= 2; y++) {
            color += texture(texture0, v_texcoord + vec2(x, y) * pixel);
        }
    }
    
    fragColor = color / 25.0;
}
'''

def draw():
    dot.background((0, 0, 0))  # Can clear freely
    dot.circle((dot.mouse_x, dot.mouse_y), 50)
    dot.apply_shader(blur_shader, accumulate=False)
```

## Tips

- **Accumulating shaders**: Avoid calling `dot.background()` in draw loop or effects will be cleared
- **Non-accumulating shaders**: Safe to call `dot.background()` - shader is just a filter
- **Chaining shaders**: Call `apply_shader()` multiple times for combined effects. Make sure you have `accumulate=True` in shaders before the last one in the chain to pass through effects 

```python
dot.apply_shader(self.pixelate, pixelSize=int(mean_amp*100), accumulate=True)
dot.apply_shader(self.rgb_split, accumulate=False, offset=mean_amp*0.3)
```

- **Performance**: Complex shaders (many texture lookups) may reduce framerate
- **Debugging**: If shader doesn't compile, check console for error messages
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

See [Examples](examples/interaction/key_press.py)

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

### Audio Sequencing Reference

Sequence and trigger audio samples with the Sampler and Clock classes.

#### Sampler

The Sampler class loads and triggers audio samples.

##### Creating a Sampler

```python
from dorothy.Audio import Sampler

def setup(self):
    self.sampler = Sampler(dot)
Loading Samples
python# Load multiple samples
paths = [
    "../audio/kick.wav",
    "../audio/snare.wav",
    "../audio/hihat.wav"
]
self.sampler.load(paths)
# Samples are indexed 0, 1, 2, etc.
```

##### Triggering Samples
```python
# Trigger by index
self.sampler.trigger(0)  # Play first sample (kick)
self.sampler.trigger(1)  # Play second sample (snare)

# Samples play polyphonically (overlapping is fine)
```

#### Clock

The Clock class provides tempo-synced timing for sequencing.

##### Creating a Clock
```python
def setup(self):
    self.clock = dot.music.get_clock()
    self.clock.set_bpm(120)  # Set tempo (default: 80)
```

###### Clock Callbacks
```python
def setup(self):
    self.clock = dot.music.get_clock()
    self.clock.set_bpm(120)
    
    # Called on every tick
    self.clock.on_tick = self.on_tick
    
    # Start the clock
    self.clock.play()

def on_tick(self):
    # This runs at each clock tick
    print(f"Tick: {self.clock.tick_ctr}")
```

##### Clock Properties
```python
self.clock.tick_ctr          # Current tick number (starts at 0, increments each tick)
self.clock.bpm               # Current BPM
self.clock.ticks_per_beat    # Subdivisions per beat (default: 4 = 16th notes)
self.clock.playing           # True if clock is running
self.clock.tick_length       # Milliseconds between ticks (calculated from BPM)
```

##### Clock Methods
###### play()
Start the clock. Resets tick_ctr to 0.
```python
self.clock.play()
```
###### stop()
Stop the clock.
```python
self.clock.stop()
```
###### set_bpm(bpm)
Set the tempo in beats per minute.
```python
self.clock.set_bpm(120)  # 120 BPM
self.clock.set_bpm(80)   # Slower
self.clock.set_bpm(180)  # Faster
```
###### set_tpb(ticks_per_beat)
Set the tick subdivision per beat.
```python
self.clock.set_tpb(4)   # 16th notes (default)
self.clock.set_tpb(2)   # 8th notes
self.clock.set_tpb(8)   # 32nd notes
self.clock.set_tpb(1)   # Quarter notes
```

Note: Call set_tpb() AFTER set_bpm() to ensure tick length is calculated correctly.
#### Sequencing Example
Create a step sequencer with samples:
```python
def setup(self):
    # Load samples
    paths = [
        "../audio/kick.wav",
        "../audio/snare.wav",
        "../audio/hihat.wav",
    ]
    self.sampler = Sampler(dot)
    self.sampler.load(paths)
    
    # Setup clock
    self.clock = dot.music.get_clock()
    self.clock.set_bpm(120)
    self.clock.on_tick = self.on_tick
    
    # Define sequence (0 = rest, 1+ = sample index + 1)
    self.sequence = [
        1, 0, 0, 0,  # Kick on 1
        2, 0, 0, 0,  # Snare on 5
        1, 0, 3, 0,  # Kick + hihat
        2, 0, 3, 0,  # Snare + hihat
    ]
    
    self.clock.play()

def on_tick(self):
    # Get current step in sequence
    step = self.clock.tick_ctr % len(self.sequence)
    note = self.sequence[step]
    
    # Trigger sample if not a rest
    if note > 0:
        self.sampler.trigger(note - 1)
```

#### Visual Sequencer Example
Draw a step sequencer with playhead:
```python
def setup(self):
    # Load samples
    paths = [
        "../audio/snare.wav",
        "../audio/snare2.wav",
        "../audio/meow.wav",
    ]
    self.sampler = Sampler(dot)
    self.sampler.load(paths)
    
    # Setup clock
    self.clock = dot.music.get_clock()
    self.clock.set_bpm(80)
    self.clock.on_tick = self.on_tick
    
    # Sequence (0 = rest, 1-3 = sample indices)
    self.sequence = [1, 0, 2, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 2, 2, 2]
    
    # Create grid positions
    self.grid = np.linspace(0, dot.width, len(self.sequence))
    
    self.clock.play()

def on_tick(self):
    n = len(self.sequence)
    note = self.sequence[self.clock.tick_ctr % n]
    
    if note > 0:
        self.sampler.trigger(note - 1)

def draw(self):
    dot.background(dot.darkblue)
    
    # Draw all steps
    y = dot.height / 2
    for x in self.grid:
        dot.fill(dot.white)
        dot.circle((x, y), 10)
    
    # Draw playhead
    n = len(self.sequence)
    x = self.grid[self.clock.tick_ctr % n]
    dot.fill(dot.red)
    dot.circle((x, y), 10)
```

#### Multiple Tracks
```python
def setup(self):
    self.sampler = Sampler(dot)
    self.sampler.load(["kick.wav", "snare.wav", "hat.wav"])
    
    self.clock = dot.music.get_clock()
    self.clock.set_bpm(120)
    self.clock.on_tick = self.on_tick
    
    # Separate patterns for each sample
    self.kick_pattern  = [1, 0, 0, 0, 1, 0, 0, 0]
    self.snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0]
    self.hat_pattern   = [1, 1, 1, 1, 1, 1, 1, 1]
    
    self.clock.play()

def on_tick(self):
    step = self.clock.tick_ctr % 8
    
    if self.kick_pattern[step]:
        self.sampler.trigger(0)
    if self.snare_pattern[step]:
        self.sampler.trigger(1)
    if self.hat_pattern[step]:
        self.sampler.trigger(2)
```

#### Tips

* BPM range: Typical range is 60-180 BPM

* Tick resolution: Default is 4 ticks per beat (16th notes). Adjust with set_tpb()
  
* Timing precision: Clock runs in a separate thread with ~1ms precision
  
* Stop before restarting: Call clock.stop() before creating a new Clock or calling play() again
  
* Polyphony: Sampler plays samples polyphonically - multiple samples can overlap
  
* File formats: Supports WAV
  
* Performance: Pre-load all samples in setup() for best performance

#### Common Patterns
4/4 Time Signature
```python
self.clock.set_tpb(4)  # 16th notes
sequence_length = 16   # 4 beats × 4 ticks
```
3/4 Time Signature (Waltz)
```python
self.clock.set_tpb(4)
sequence_length = 12   # 3 beats × 4 ticks
```

### Stream Samples 

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

### RAVE Model Generation

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

### MAGNet Model Generation

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

### Smoothing Signals

#### get_window(window_size, method = "average", dims = 1)

Make a window for smoothing signals 

#### add(val or List)

Add new value, returns current smoothed value 

#### Example

```python
w = dot.get_window(10)
mean = w.add(new_val)
mean = w.add(new_val)
mean = w.add(new_val)
...
```

```python
w = dot.get_window(10, dims = 3)
mean = w.add([x,y,z])
mean = w.add([x,y,z])
mean = w.add([x,y,z])
...
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

See [Examples](examples/audio_playback/multi_audio_outputs.py)

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

See [Examples](examples)

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

See [Examples](examples)

## Tips & Best Practices

### Performance

1. **Minimize state changes** - batch drawing with same colors
2. **Use layers for static content** - draw once, reuse many times
3. **Cache audio data** - read `fft_vals` once per frame
4. **Reduce geometry complexity** - subsample data when possible

### Transforms

1. **Order matters** - translate → rotate → scale is typical
2. **Scale from center** - translate to center, scale, translate back

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

- Use `with dot.transforms:`
- Is your drawing code within the block?
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



## Credits

Dorothy by Louis McCallum  
ModernGL refactor maintains API compatibility while adding GPU acceleration and 3D support.