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


#### text (text, x, y, size)

**Parameters:**
- `text` (str): Text to render
- `x` (int): x coordinate
- `y` (int): y coordinate
- `size` (int): font size

**Example:**

```python
dot.text("hello world"), 100, 200, 36)
```

### 3D Shapes

#### camera_3d()

Must be called before drawing anything in 3d. Call `camera_2d()` to go back to 2D

#### set_camera(eye=(1,1,1), target = (0,0,0))
**Parameters:**
- `eye` (Tuple): Where the camera is
- `target` (Tuple): What its looking at

**Example:**
```python
angle = dot.frames * 0.01
x = 5 * math.cos(angle)
z = 5 * math.sin(angle)
dot.set_camera((x, 2, z), (0, 0, 0))
```

#### dot.renderer.light_pos = (x, y, z)

#### use_lighting(bool=True)

**Example:**
```python
if dot.frames%20==0:
    dot.use_lighting(not dot.renderer.use_lighting)
```

#### sphere(radius=1.0, pos = (0,0,0))

Draw a 3D sphere (requires 3D camera mode).

**Parameters:**
- `radius` (float): Sphere radius (default: 1.0)
- `pos` (Tuple): position of sphere (default: 0,0,0)

**Example:**
```python
dot.camera_3d()
dot.set_camera((0, 0, 5), (0, 0, 0))
dot.fill((255, 100, 100))
dot.sphere(1.0)
```

#### box(size=(1,1,1), pos = (0,0,0), texture_layers= int or {} )

Draw a 3D box (requires 3D camera mode).

**Parameters:**
- `size` (Tuple): size of sphere (default: 1,1,1)
- `pos` (Tuple): position of sphere (default: 0,0,0)
- `texture_layers` (dict or layer): A pointer to a layer for all 6 sides (stretched to fit), or a dictionary of layers for each side 

**Example:**
```python
dot.camera_3d()
dot.fill((100, 100, 255))
dot.box((2.0, 1.0, 1.5))
```

**Example:**
```python
dot.camera_3d()
dot.fill((100, 100, 255))
dot.box((2.0, 1.0, 1.5),(10,10,2))
```

```python
dot.box((20, 20, 20),texture_layers = self.front_layer)
```

```python
dot.box((20, 20, 20), 
    texture_layers={
    'front': self.front_layer,
    'back': self.back_layer,
    'right': self.right_layer,
    'left': self.left_layer,
    'top': self.top_layer,
    'bottom': self.bottom_layer
})
```


#### line_3d(pos1, pos2)

Draw a 3D line (requires 3D camera mode).

**Parameters:**
- `pos1` ([x, y, z]): start point
- `pos2` ([x, y, z]): end point

**Example:**
```python
dot.camera_3d()
dot.stroke((100, 100, 255))
dot.line_3d((0, 0,0), (2, 2,2))
```

#### polyline_3d([pos1...n], closed = True)

Draw a 3D polyline (requires 3D camera mode).

**Parameters:**
- `pos` ([(x, y, z),(x, y, z),....]): Array of 3d coordinates
- `closed` (bool): Is shaped closed? Defaults to True

**Example:**
```python
dot.camera_3d()
dot.stroke((100, 100, 255))
dot.line_3d((0, 0,0), (2, 2,2))
```

#### .obj files

Load and texture a `.obj` file

##### load_obj(filepath)

##### draw_mesh(obj, texture)

##### Example

```python
def setup(self):
    self.tree = dot.load_obj("model/Tree1.obj")
    self.texture_layer = dot.get_layer()

def draw(self):
    dot.background(dot.black)
    with dot.layer(self.texture_layer):
        dot.camera_2d()
        x = (dot.frames * 5) % dot.width
        dot.circle((x, dot.height//2), 50)
    
    # Draw mesh
    dot.camera_3d()
    dot.draw_mesh(self.tree, self.texture_layer)
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

#### `pixelate(pixel_size, bake)`
Pixelate the canvas into larger blocks.
```python
dot.pixelate(pixel_size=8.0, bake=False)
```

**Parameters:**
- `pixel_size` (float): Size of pixel blocks. Larger = more pixelated (default: 8.0)
- `bake` (bool): If True, writes effect back into the canvas. If False, overlays as a display filter (default: False)

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 50)
dot.pixelate(12.0)
```

---

#### `blur(bake)`
Apply blur effect to the canvas.
```python
dot.blur(bake=False)
```

**Parameters:**
- `bake` (bool): If True, writes effect back into the canvas. If False, overlays as a display filter (default: False)

**Example:**
```python
dot.circle((200, 200), 100)
dot.blur()
```

---

#### `rgb_split(offset, bake)`
Split RGB channels for glitch/chromatic aberration effect.
```python
dot.rgb_split(offset=0.01, bake=False)
```

**Parameters:**
- `offset` (float): Distance to split channels, range 0.0-0.1 (default: 0.01)
- `bake` (bool): If True, writes effect back into the canvas. If False, overlays as a display filter (default: False)

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 50)
dot.rgb_split(0.02)
```

---

#### `feedback(zoom, bake)`
Create feedback/zoom effect with trails.
```python
dot.feedback(zoom=0.98, bake=True)
```

**Parameters:**
- `zoom` (float): Zoom factor per frame. <1.0 zooms out, >1.0 zooms in (default: 0.98)
- `bake` (bool): If True, writes effect back into the canvas so it compounds each frame (default: True)

**Example:**
```python
dot.circle((dot.mouse_x, dot.mouse_y), 20)
dot.feedback(0.99)  # Slow zoom out with trails
```

---

#### `roll(offset_x, offset_y, bake)`
Scroll/shift the canvas with wrapping (like `np.roll`).
```python
dot.roll(offset_x=0.0, offset_y=0.0, bake=True)
```

**Parameters:**
- `offset_x` (float): Horizontal shift in pixels. Positive = right (default: 0.0)
- `offset_y` (float): Vertical shift in pixels. Positive = down (default: 0.0)
- `bake` (bool): If True, writes effect back into the canvas so the shift compounds each frame (default: True)

**Example:**
```python
dot.circle((dot.width//2, dot.height//2), 50)
dot.roll(2.0, 0.0)  # Scroll right continuously
```

---

#### `invert(bake)`
Invert all colors on the canvas.
```python
dot.invert(bake=False)
```

**Parameters:**
- `bake` (bool): If True, writes effect back into the canvas. If False, overlays as a display filter (default: False)

**Example:**
```python
dot.background(dot.white)
dot.fill(dot.red)
dot.circle((200, 200), 100)
dot.invert()  # White becomes black, red becomes cyan
```

---

#### `tile(grid_x, grid_y, bake)`
Tile/repeat the canvas in a grid pattern.
```python
dot.tile(grid_x=2, grid_y=2, bake=False)
```

**Parameters:**
- `grid_x` (int): Number of horizontal tiles (default: 2)
- `grid_y` (int): Number of vertical tiles (default: 2)
- `bake` (bool): If True, writes effect back into the canvas. If False, overlays as a display filter (default: False)

**Example:**
```python
dot.circle((100, 100), 50)
dot.tile(4, 4)  # Create 4x4 grid of circles
```

---

#### `cutout(color, threshold, bake)`
Make pixels of a specific color transparent (chroma key/green screen).
```python
dot.cutout(color, threshold=0.1, bake=True)
```

**Parameters:**
- `color` (tuple or constant): RGB color to cut out, e.g. `(0, 0, 0)` or `dot.green`
- `threshold` (float): Color matching tolerance. 0.0 = exact, 0.5 = loose (default: 0.1)
- `bake` (bool): If True, writes effect back into the canvas (default: True)

**Example:**
```python
dot.background(dot.black)
dot.fill(dot.red)
dot.circle((200, 200), 100)
dot.cutout((0, 0, 0))  # Remove black background
```

---

### Baked vs Display Filter

**Baked (`bake=True`)**: Effect is written back into the persistent canvas. Without `background()`, subsequent frames build on the result — use for feedback effects, trails, and chaining effects together.

**Display filter (`bake=False`)**: Effect is overlaid on screen only. The canvas is unchanged — use for post-processing like blur, pixelate, color correction.

**Example - Baked (feedback loop):**
```python
def draw(self):
    dot.circle((dot.mouse_x, dot.mouse_y), 20)
    dot.feedback(0.99, bake=True)  # Creates trails
```

**Example - Display filter:**
```python
def draw(self):
    dot.background(dot.white)
    dot.circle((200, 200), 50)
    dot.pixelate(8.0, bake=False)  # Just a visual filter
```

---

### Combining Effects

Chain multiple effects together. Use `bake=True` for all effects except the last so each result passes through to the next:
```python
def draw(self):
    dot.circle((dot.mouse_x, dot.mouse_y), 30)
    
    dot.feedback(0.98, bake=True)   # Zoom trails
    dot.rgb_split(0.015, bake=True) # Glitch
    dot.pixelate(6.0)               # Retro look (bake=False is fine for the last effect)
```

---


## Custom Shaders Reference

Apply custom GLSL fragment shaders to create visual effects and post-processing.

## Basic Usage

```python
dot.apply_shader(shader_code, bake=True, **uniforms)
```

**Parameters:**
- `shader_code` (str): GLSL fragment shader source code
- `bake` (bool):
  - `True`: Writes shader result back into the canvas. Without `background()`, effects compound each frame (feedback effects).
  - `False`: Shader is a display-only filter, canvas unchanged (post-processing)
- `**uniforms`: Additional uniforms to pass to shader (e.g., `time=1.5`, `amount=0.1`)

## Shader Template

```glsl
#version 330

uniform sampler2D texture0;  // The canvas texture (always available)
uniform vec2 resolution;      // Canvas size in pixels (optional)
uniform float my_parameter;   // Add any user defined paramters (uniforms)

in vec2 v_texcoord;          // Texture coordinates (0-1)
out vec4 fragColor;          // Output color

void main() {
    vec4 color = texture(texture0, v_texcoord);
    // Modify color here
    fragColor = color;
}
```

```python
dot.apply_shader(shader_code, bake=True, my_parameter=1.0)
```


## Baked vs Display Filter

### Baked (`bake=True`)
Shader output is written back into the persistent canvas.

**Use for:** Feedback effects, trails, decay, generative art. Without `background()`, each frame builds on the last.

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
    # Don't call background() - canvas accumulates history
    dot.circle((dot.mouse_x, dot.mouse_y), 20)
    dot.apply_shader(feedback_shader, bake=True, fade=0.98)
```

### Display filter (`bake=False`)
Shader output is displayed but the canvas is unchanged. No feedback.

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
    dot.background((0, 0, 0))  # Safe to clear - shader doesn't touch the canvas
    dot.circle((dot.mouse_x, dot.mouse_y), 50)
    dot.apply_shader(blur_shader, bake=False)
```

## Tips

- **Baked shaders**: Avoid calling `dot.background()` in draw loop or the canvas will be wiped each frame
- **Display filter shaders**: Safe to call `dot.background()` - shader is just an overlay
- **Chaining shaders**: Call `apply_shader()` multiple times for combined effects. Use `bake=True` on all but the last so each result passes through to the next.

```python
dot.apply_shader(self.pixelate, pixelSize=int(mean_amp*100), bake=True)
dot.apply_shader(self.rgb_split, bake=False, offset=mean_amp*0.3)
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

### Buttons!

**Example:**
```python
def setup(self):
    # Create a simple button
    def on_button_click(btn):
        print(f"Button '{btn.text}' was clicked!")
    
    def on_hover(btn):
        print(f"Button '{btn.text}' was hovered!")
    
    dot.create_button(300, 250, 200, 50, 
                    text="Click Me",
                    id="button1",
                    on_release=on_button_click, on_hover=on_hover)

def draw(self):
    dot.background((40, 40, 50))
    
    # Update and draw buttons
    dot.update_buttons()
    dot.draw_buttons()
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

`Note`, `Sequence`, `Clock`, and audio devices (`PolySynth`, `Sampler`, `GranularSynth`) form a unified sequencing system. A `Sequence` is connected to a `Clock` and an audio device; the clock drives the sequence which fires `note_on` / `note_off` on the device.

```python
from dorothy.Audio import Sequence, Note

def setup(self):
    self.clock = dot.music.get_clock(bpm=120)
    self.clock.set_tpb(4)                        # 4 ticks per beat

    idx = dot.music.start_poly_synth_stream()
    self.synth = dot.music.audio_outputs[idx]

    self.seq = Sequence(steps=8, ticks_per_step=4)
    self.seq[0] = Note(60)
    self.seq.connect(self.clock, self.synth)
    self.clock.play()
```

---

#### Note

`Note` is a dataclass representing a single step event.

```python
Note(
    midi,            # MIDI note number (0-127). Middle C = 60, A4 = 69
    vel=0.8,         # Velocity 0.0–1.0
    duration=1,      # Duration in steps before note_off fires
    # Per-note ADSR overrides (None = use device default)
    attack=None,
    decay=None,
    sustain=None,
    release=None,
    # Per-note oscillator overrides (PolySynth only)
    waveform=None,   # 'sine'|'saw'|'triangle'|'noise'|'supersaw'|'fm'|'pwm'
    fm_ratio=None,
    fm_index=None,
    detune=None,
    n_oscs=None,
    pwm=None,
)
```

```python
note.freq   # Read-only: MIDI → Hz (440 * 2**((midi-69)/12))
```

---

#### Clock

Provides tempo-synced timing. Runs in a background thread.

##### Creating a Clock
```python
self.clock = dot.music.get_clock(bpm=120)
self.clock.set_tpb(4)   # ticks per beat (default 4)
```

##### Properties
```python
self.clock.tick_ctr       # Current tick count (increments before callbacks fire)
self.clock.bpm            # Current BPM
self.clock.ticks_per_beat # Subdivisions per beat
self.clock.playing        # True if running
self.clock.tick_length    # Milliseconds per tick
```

##### Methods
```python
self.clock.play()              # Start (resets tick_ctr to 0)
self.clock.stop()              # Stop
self.clock.set_bpm(120)        # Change tempo
self.clock.set_tpb(4)         # Change tick subdivision

# Register callbacks — multiple callbacks are supported
self.clock.on_tick_fns.append(self.my_fn)
```

##### Timing grid
```python
# 4/4 — 16th-note steps
self.clock.set_tpb(4)   # 4 ticks/beat → ticks_per_step=1 → 16th notes
                         #              → ticks_per_step=4 → quarter notes

# Tip: set_tpb() after set_bpm() so tick_length recalculates correctly
```

---

#### Sequence

Step sequencer that drives any compatible audio device.

##### Creating and connecting
```python
seq = Sequence(steps=16, ticks_per_step=1)
seq.connect(clock, synth)   # registers tick callback; call before clock.play()
```

##### Step editing
```python
seq[i] = Note(60)              # single note
seq[i] = [Note(60), Note(64)]  # chord
seq[i] = []                    # rest
note = seq[i]                  # read a step

seq.steps = 32                 # resize (current_step wraps into new range)
seq.ticks_per_step = 2         # change step resolution live
```

##### Pattern methods
```python
seq.clear()          # empty all steps
seq.clear(i)         # empty one step
seq.set_pattern([    # replace all steps atomically; sends all_notes_off first
    [Note(60)],
    [],
    [Note(64), Note(67)],
    [],
])
seq.all_notes_off()  # immediately release all pending notes
```

---

#### PolySynth

Polyphonic synthesizer AudioDevice. Up to `n_voices` simultaneous notes.

##### Creating
```python
idx = dot.music.start_poly_synth_stream(
    n_voices=8,
    n_harmonics=4,
    attack=0.01,    decay=0.1,    sustain=0.7,    release=0.3,
    waveform='sine',             # default oscillator shape
    buffer_size=512,
    sr=44100,
)
synth = dot.music.audio_outputs[idx]
```

##### Waveforms
`'sine'` · `'saw'` · `'triangle'` · `'noise'` · `'supersaw'` · `'fm'` · `'pwm'`

##### Default parameters (read/write)
```python
synth.attack        # ADSR attack (seconds)
synth.decay         # ADSR decay (seconds)
synth.sustain       # ADSR sustain level 0–1
synth.release       # ADSR release (seconds)
synth.waveform      # Default oscillator shape
synth.fm_ratio      # FM: modulator = fm_ratio × carrier (default 2.0)
synth.fm_index      # FM: modulation depth in radians (default 1.0)
synth.detune        # Supersaw: total semitone spread (default 0.2)
synth.n_oscs        # Supersaw: oscillator count (default 7)
synth.pwm           # PWM: duty cycle 0–1 (default 0.5 = square)
```

##### Direct API (thread-safe)
```python
synth.note_on(freq, vel=0.8, waveform='saw', attack=0.05, ...)
synth.note_off(freq)
synth.all_notes_off()
```

Per-note overrides in `note_on` apply to that note only; `None` falls back to the synth default. Notes passed through a `Sequence` carry overrides from their `Note` fields.

---

#### Sampler

Sample player AudioDevice. `Note.midi` is used as the slot index; `Note.vel` scales volume.

##### Creating
```python
idx = dot.music.start_sampler_stream(
    paths=["kick.wav", "snare.wav", "hat.wav"],  # optional pre-load
    sr=44100,
    buffer_size=512,
)
sampler = dot.music.audio_outputs[idx]
sampler.load(["kick.wav", "snare.wav"])   # load or swap samples at any time
```

Slot 0 = `paths[0]`, slot 1 = `paths[1]`, etc.

##### Sequence usage
```python
seq[0] = Note(0, vel=1.0)   # trigger slot 0
seq[2] = Note(1, vel=0.8)   # trigger slot 1
seq.connect(clock, sampler)
```

Samples play to their natural end; `note_off` is a no-op (one-shots always complete).

##### Direct API (thread-safe)
```python
sampler.trigger(0, vel=1.0)   # trigger by slot index directly
sampler.all_notes_off()        # stop all playing voices immediately
```

---

#### GranularSynth

Granular synthesis AudioDevice. Loads one audio file and plays it as overlapping short grains.

`Note.midi` 69 (A4, 440 Hz) = original file pitch. Other values shift pitch by semitone distance from A4. `Note.vel` scales voice volume.

##### Creating
```python
idx = dot.music.start_granular_stream(
    path="texture.wav",     # optional pre-load
    position=0.5,           # initial read position 0–1
    spread=0.1,
    grain_size=80.0,        # ms
    density=8.0,            # grains/sec/voice
    attack=0.3,    decay=0.3,
    n_grains=32,
    pitch=0.0,              # semitones
    pitch_spread=0.0,       # per-grain jitter (semitones std dev)
    sr=44100,
    buffer_size=512,
)
gran = dot.music.audio_outputs[idx]
gran.load("texture.wav")    # load or swap source at any time
```

##### Parameters (read/write at any time)
```python
gran.position      # 0–1, read head centre in source file
gran.spread        # 0–1, random position scatter (fraction of file)
gran.grain_size    # ms per grain
gran.density       # grains per second per active voice
gran.attack        # fraction of grain for fade-in
gran.decay         # fraction of grain for fade-out
gran.n_grains      # max simultaneous grains
gran.pitch         # global semitone shift
gran.pitch_spread  # per-grain pitch jitter (Gaussian std dev, semitones)
```

##### Direct API (thread-safe)
```python
gran.note_on(freq, vel=0.8)   # start grain cloud at pitch/volume
gran.note_off(freq)            # stop spawning; active grains play out
gran.all_notes_off()           # silence immediately, clear all grains
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
#### is_onset(output=0)

Detect if an onset occurred since last call. This works offline (if playing back audio loaded in at the beginning), or online in a streaming fashion (although this is less accurate)

```python
o = dot.music.start_device_stream(1)
dot.music.audio_outputs[o].onset_detector.threshold = 0.5 
dot.music.audio_outputs[o].analyse_onsets = True
```

**Parameters:**
- `output` (int): Audio output index (default: 0)

**Returns:**
- `bool`: True if onset detected

**Example:**
```python
if dot.music.is_onset():
    dot.fill((255, 0, 0))
```

#### is_beat(output=0)

Detect if a beat occurred since last call. This works offline (if playing back audio loaded in at the beginning), or online in a streaming fashion (although this is less accurate)

```python
o = dot.music.start_device_stream(1)
dot.music.audio_outputs[o].onset_detector.threshold = 0.5 
dot.music.audio_outputs[o].analyse_onsets = True
dot.music.audio_outputs[o].analyse_beats = True
```

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

You can set up Dorothy to update the sketch everytime you save the file

The main changes are 

1. No `__init__()` function in the `MySketch` class

2. Don't make an instance of the `MySketch` class

3. Instead, run the `start_livecode_loop()` function


```python
if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
```

### Example

```python
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    def setup(self):
        self.col = (0,255,0)
        print("start")

    def run_once(self):
        print("run once")
        self.col = (0,0,0)
                
    def draw(self):
        dot.background(self.col)
        dot.fill(dot.blue)
        dot.rectangle((0,dot.frames%40),(400,100))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
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