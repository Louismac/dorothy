# Dorothy API Reference

Dorothy is a creative computing Python library with a ModernGL (GPU-accelerated) backend. It's built for sketches, live visuals, audio-reactive art, and live coding — think Processing/p5.js but in Python, with first-class audio, 3D, and shader support.

## Quick start

```bash
pip install dorothy-cci
```

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

## Cheat sheet

The most-used APIs, grouped by what you're trying to do. Follow the links for full details.

### Drawing
| Signature | Does |
|---|---|
| `dot.background(color)` | Clear screen to a color |
| `dot.fill(color)` / `dot.no_fill()` | Set / disable shape fill |
| `dot.stroke(color)` / `dot.no_stroke()` | Set / disable shape outline |
| `dot.set_stroke_weight(w)` | Line width in pixels |
| `dot.circle(center, radius)` | Draw a circle |
| `dot.rectangle(pos1, pos2)` | Draw a rectangle |
| `dot.line(pos1, pos2)` | Draw a line |
| `dot.polyline(pts, closed=False)` | Multi-point line |
| `dot.polygon(pts)` | Filled polygon |
| `dot.text(txt, x, y, size)` | Render text |

### Animation & state
| Signature | Does |
|---|---|
| `dot.frames` | Frame counter |
| `dot.millis` | Milliseconds since start |
| `dot.mouse_x`, `dot.mouse_y` | Current mouse position |
| `dot.get_lfo(osc, freq, range)` | Create an oscillator |
| `dot.lfo_value(id)` | Read its current value |

### Transforms
| Signature | Does |
|---|---|
| `with dot.transform():` | Scope transforms to a block |
| `dot.translate(x, y, z=0)` | Move origin |
| `dot.rotate(angle, x=0, y=0, z=1)` | Rotate (radians) |
| `dot.scale(s, y=None, z=None)` | Scale |

### Effects & images
| Signature | Does |
|---|---|
| `dot.paste(img, position)` | Draw a NumPy image |
| `dot.blur()` / `dot.pixelate(n)` / `dot.invert()` | Built-in effects |
| `dot.feedback(zoom)` | Zoom-trail feedback |
| `dot.apply_shader(code, bake=...)` | Custom GLSL shader |

### Audio
| Signature | Does |
|---|---|
| `dot.music.start_file_stream(path)` | Play audio file |
| `dot.music.amplitude()` | Current volume |
| `dot.music.fft()` | Current FFT spectrum |
| `dot.music.is_beat()` / `is_onset()` | Beat / onset detection |

### Lifecycle
| Signature | Does |
|---|---|
| `dot.start_loop(setup, draw)` | Standard main loop |
| `dot.start_livecode_loop(module)` | Hot-reload on file save |

---

## Table of contents

**Getting started**
- [Installation & first sketch](#installation--first-sketch)
- [The Dorothy class](#the-dorothy-class)

**Drawing**
- [2D shapes](#2d-shapes)
- [3D shapes](#3d-shapes)
- [Style: fill, stroke, background](#style-fill-stroke-background)
- [Text](#text)
- [Cameras (2D/3D)](#cameras)

**Animation, input, and state**
- [Properties (frames, mouse, time)](#properties)
- [LFOs / oscillators](#lfos--oscillators)
- [Interaction callbacks](#interaction-callbacks)

**Composition**
- [Transforms](#transforms)
- [Layers](#layers)
- [Images](#images)

**Effects & shaders**
- [Built-in effects](#built-in-effects)
- [Baked vs display filter](#baked-vs-display-filter)
- [Custom GLSL shaders](#custom-glsl-shaders)

**Audio** 
- [Audio sources](#audio-sources) — file, device (mic), DSP callback, samples, RAVE, MAGNet
- [Analysis](#audio-analysis) — amplitude, FFT, onset, beat detection
- [Signal smoothing](#signal-smoothing)
- [Playback control](#playback-control)
- [Sequencing](#audio-sequencing) — Clock, Sequence, Note
- [Instruments](#instruments) — PolySynth, Sampler, GranularSynth
- [Advanced](#advanced-audio-features) — RAVE timbre transfer, callbacks, gain
- [Performance tips](#audio-performance-tips)
- [Troubleshooting](#audio-troubleshooting)
  
**Live coding**
- [Hot-reload loop](#live-coding)

**Appendix**
- [Color constants](#color-constants)
- [Key constants](#key-constants)
- [Troubleshooting](#troubleshooting)
- [Tips & best practices](#tips--best-practices)
- [Version history](#version-history)

---

# Getting started

## Installation & first sketch

```bash
pip install dorothy-cci
```

See the [Quick start](#quick-start) above for a minimal runnable sketch.

## The Dorothy class

The main class for creating Dorothy applications. A single instance (conventionally named `dot`) holds the window, renderer, and audio system.

### Constructor

```python
Dorothy(width=800, height=600, title="Dorothy")
```

**Parameters**
- `width` *(int)* — Window width in pixels.
- `height` *(int)* — Window height in pixels.
- `title` *(str)* — Window title.

**Example**
```python
dot = Dorothy(width=1920, height=1080, title="My Project")
```

### `start_loop(setup_fn, draw_fn)`

Start the main render loop.

**Parameters**
- `setup_fn` *(Callable)* — Runs once at startup.
- `draw_fn` *(Callable)* — Runs every frame (~60fps).

**Example**
```python
def setup():
    dot.background((255, 255, 255))

def draw():
    dot.circle((dot.mouse_x, dot.mouse_y), 20)

dot.start_loop(setup, draw)
```

### `start_livecode_loop(sketch_module)`

Start a live coding loop that reloads your sketch on file save. See [Live coding](#live-coding) for the full pattern.

**Parameters**
- `sketch_module` — Python module containing a `MySketch` class.

---

# Drawing

All drawing functions respect the current fill, stroke, and transform state. Set fill/stroke first, then draw.

## 2D shapes

### `circle(center, radius, annotate=False)`

Draw a circle.

**Parameters**
- `center` *(Tuple[float, float])* — `(x, y)` center position.
- `radius` *(float)* — Circle radius.
- `annotate` *(bool)* — Draw a debug cross at the center.

**Example**
```python
dot.fill((255, 0, 0))
dot.stroke((0, 0, 0))
dot.set_stroke_weight(2)
dot.circle((400, 300), 100)

# With annotation for debugging
dot.circle((200, 200), 50, annotate=True)
```

### `rectangle(pos1, pos2, annotate=False)`

Draw a rectangle.

**Parameters**
- `pos1` *(Tuple[float, float])* — `(x1, y1)` top-left corner.
- `pos2` *(Tuple[float, float])* — `(x2, y2)` bottom-right corner.
- `annotate` *(bool)* — Draw a debug marker.

**Example**
```python
dot.fill((0, 255, 0))
dot.rectangle((100, 100), (300, 200))
```

### `line(pos1, pos2, annotate=False)`

Draw a line.

**Parameters**
- `pos1` *(Tuple[float, float])* — `(x1, y1)` start point.
- `pos2` *(Tuple[float, float])* — `(x2, y2)` end point.
- `annotate` *(bool)* — Draw a debug marker.

**Example**
```python
dot.stroke((255, 255, 0))
dot.set_stroke_weight(5)
dot.line((0, 0), (800, 600))
```

### `polyline(points, closed=False)`

Draw a multi-point line.

**Parameters**
- `points` *(List[Tuple[float, float]])* — Points of the line.
- `closed` *(bool)* — Close the line into a shape (first point connected to last).

**Example**
```python
# Star shape (concave)
import numpy as np
points = []
for i in range(10):
    angle = i * np.pi / 5 - np.pi / 2
    r = 100 if i % 2 == 0 else 50
    x = 400 + r * np.cos(angle)
    y = 300 + r * np.sin(angle)
    points.append((x, y))

dot.stroke(dot.red)
dot.set_stroke_weight(10)
dot.polyline(points, closed=True)
```

### `polygon(points)`

Draw a filled polygon.

**Parameters**
- `points` *(List[Tuple[float, float]])* — Points of the polygon.

**Example**
```python
# Same star as above, filled
dot.fill(dot.red)
dot.polygon(points)
```

## Text

### `text(text, x, y, size)`

Render a text string.

**Parameters**
- `text` *(str)* — Text to render.
- `x` *(int)* — X coordinate.
- `y` *(int)* — Y coordinate.
- `size` *(int)* — Font size.

**Example**
```python
dot.text("hello world", 100, 200, 36)
```

## Style: fill, stroke, background

### `fill(color)`

Set fill color for shapes.

**Parameters**
- `color` *(Tuple[int, int, int] or Tuple[int, int, int, int])* — RGB or RGBA color (0–255).

**Example**
```python
dot.fill((255, 0, 0))        # Red
dot.fill((0, 255, 0, 128))   # Green, 50% transparent
```

### `no_fill()`

Disable fill for shapes (only stroke will be drawn).

```python
dot.no_fill()
dot.stroke((0, 0, 0))
dot.circle((400, 300), 50)  # Hollow circle
```

### `stroke(color)`

Set stroke color for shape outlines.

**Parameters**
- `color` *(Tuple[int, int, int] or Tuple[int, int, int, int])* — RGB or RGBA color (0–255).

```python
dot.stroke((0, 0, 255))
dot.set_stroke_weight(3)
```

### `no_stroke()`

Disable stroke for shapes (only fill will be drawn).

```python
dot.fill((255, 0, 0))
dot.no_stroke()
dot.circle((400, 300), 50)  # Filled circle, no outline
```

### `set_stroke_weight(weight)`

Set stroke line width.

**Parameters**
- `weight` *(float)* — Line width in pixels.

```python
dot.set_stroke_weight(5)
dot.line((0, 0), (100, 100))
```

### `background(color)`

Clear the screen to a color. Call at the start of `draw()` to reset the canvas each frame.

**Parameters**
- `color` *(Tuple[int, int, int])* — RGB color (0–255).

```python
dot.background((30, 30, 40))  # Dark gray
```

> ⚠️ **Note:** If you're using baked effects or shaders (trails, feedback), *don't* call `background()` every frame — it will erase the accumulated canvas. See [Baked vs display filter](#baked-vs-display-filter).

## 3D shapes

All 3D drawing requires 3D camera mode. Call [`camera_3d()`](#cameras) before drawing 3D primitives, and [`camera_2d()`](#cameras) to return to flat drawing.

### `sphere(radius=1.0, pos=(0, 0, 0))`

Draw a 3D sphere.

**Parameters**
- `radius` *(float)* — Sphere radius.
- `pos` *(Tuple)* — Position `(x, y, z)`.

**Example**
```python
dot.camera_3d()
dot.set_camera((0, 0, 5), (0, 0, 0))
dot.fill((255, 100, 100))
dot.sphere(1.0)
```

### `box(size=(1, 1, 1), pos=(0, 0, 0), texture_layers=None)`

Draw a 3D box.

**Parameters**
- `size` *(Tuple)* — Box dimensions `(x, y, z)`.
- `pos` *(Tuple)* — Box position.
- `texture_layers` *(layer_id or dict)* — A single layer applied to all 6 sides, or a dict mapping face names (`'front'`, `'back'`, `'left'`, `'right'`, `'top'`, `'bottom'`) to layers. See [Layers](#layers).

**Examples**
```python
# Plain colored box
dot.camera_3d()
dot.fill((100, 100, 255))
dot.box((2.0, 1.0, 1.5))
```

```python
# Same layer on every face
dot.box((20, 20, 20), texture_layers=self.front_layer)
```

```python
# Different layer per face
dot.box((20, 20, 20), texture_layers={
    'front':  self.front_layer,
    'back':   self.back_layer,
    'right':  self.right_layer,
    'left':   self.left_layer,
    'top':    self.top_layer,
    'bottom': self.bottom_layer,
})
```

### `line_3d(pos1, pos2)`

Draw a 3D line.

**Parameters**
- `pos1` *(Tuple[float, float, float])* — Start point.
- `pos2` *(Tuple[float, float, float])* — End point.

```python
dot.camera_3d()
dot.stroke((100, 100, 255))
dot.line_3d((0, 0, 0), (2, 2, 2))
```

### `polyline_3d(points, closed=True)`

Draw a 3D polyline.

**Parameters**
- `points` *(List[Tuple[float, float, float]])* — Array of 3D coordinates.
- `closed` *(bool)* — Connect last point to first.

```python
dot.camera_3d()
dot.stroke((100, 100, 255))
dot.polyline_3d([(0, 0, 0), (2, 2, 2), (2, 0, 2)], closed=True)
```

### Lighting

Enable per-fragment lighting for 3D surfaces.

```python
dot.use_lighting(True)                     # toggle on
dot.renderer.light_pos = (10, 10, 10)      # move the light

# Toggle at runtime
if dot.frames % 20 == 0:
    dot.use_lighting(not dot.renderer.use_lighting)
```

### Loading `.obj` meshes

```python
mesh = dot.load_obj(filepath)       # returns a mesh object
dot.draw_mesh(mesh, texture_layer)  # draw with a layer as its texture
```

**Example** — render a mesh textured by a layer you drew into:
```python
def setup(self):
    self.tree = dot.load_obj("model/Tree1.obj")
    self.texture_layer = dot.get_layer()

def draw(self):
    dot.background(dot.black)
    with dot.layer(self.texture_layer):
        dot.camera_2d()
        x = (dot.frames * 5) % dot.width
        dot.circle((x, dot.height // 2), 50)

    dot.camera_3d()
    dot.draw_mesh(self.tree, self.texture_layer)
```

## Cameras

### `camera_2d()`

Switch to 2D orthographic camera (the default). Use this to return to flat drawing after a 3D block.

```python
dot.camera_2d()
dot.circle((400, 300), 50)
```

### `camera_3d()`

Switch to 3D perspective camera. Required before drawing any 3D primitive.

```python
dot.camera_3d()
dot.set_camera((0, 0, 5), (0, 0, 0))
dot.sphere(1.0)
```

### `set_camera(eye, target=(0, 0, 0))`

Set the 3D camera position and look-at target.

**Parameters**
- `eye` *(Tuple[float, float, float])* — Camera position.
- `target` *(Tuple[float, float, float])* — Point to look at.

**Example** — orbit the origin:
```python
import math
angle = dot.frames * 0.01
x = 5 * math.cos(angle)
z = 5 * math.sin(angle)
dot.set_camera((x, 2, z), (0, 0, 0))
```

---

# Animation, input, and state

## Properties

All properties are read-only and updated automatically each frame.

| Property | Type | Meaning |
|---|---|---|
| `dot.mouse_x` | `int` | Current mouse X coordinate |
| `dot.mouse_y` | `int` | Current mouse Y coordinate |
| `dot.width` | `int` | Window width in pixels |
| `dot.height` | `int` | Window height in pixels |
| `dot.frames` | `int` | Frame count since start |
| `dot.millis` | `float` | Milliseconds since start |

**Examples**
```python
# Mouse-following circle
dot.circle((dot.mouse_x, dot.mouse_y), 20)

# Frame-driven animation
import math
x = 400 + 200 * math.sin(dot.frames * 0.05)
dot.circle((x, 300), 50)

# Time-driven animation (frame-rate independent)
t = dot.millis / 1000
x = 400 + 200 * math.sin(t * 2)
dot.circle((x, 300), 50)
```

## LFOs / oscillators

Modulate parameters with cyclical oscillators. Handy for smooth animation without hand-writing `sin` calls.

### `get_lfo(osc, freq, range)`

Create an LFO. It updates automatically each frame; read its current value with `lfo_value()`.

**Parameters**
- `osc` *(str)* — One of `'sine'`, `'saw'`, `'square'`, `'triangle'`.
- `freq` *(float)* — Frequency in Hz (cycles per second).
- `range` *(tuple)* — `(min, max)` — output is mapped to this range.

**Returns:** *(int)* LFO ID to pass to `lfo_value()`.

```python
lfo_id = dot.get_lfo(osc='sine', freq=1.0, range=(0, 1))
```

### `lfo_value(lfo_id)`

Get the current value of an LFO.

```python
value = dot.lfo_value(lfo_id)
```

### Oscillator shapes

- `'sine'` — smooth wave.
- `'saw'` — linear ramp up, sharp drop.
- `'square'` — alternates between min and max.
- `'triangle'` — linear ramp up and down.

### Tips

- **Frequency**: `0.5` = slow (2s per cycle); `2.0` = fast (0.5s per cycle).
- **Ranges** can be anything — positions, sizes, colors, angles.
- **Multiple LFOs** can run at once, each with its own frequency, for complex motion.
- **Dynamic modulation**: change an LFO's frequency or range at runtime via `dot.lfos[lfo_id]['freq']`.

## Interaction callbacks

Assign a function to any of these hooks to react to input events.

### Mouse

```python
dot.on_mouse_press   = fn(x, y, button)
dot.on_mouse_release = fn(x, y, button)
dot.on_mouse_drag    = fn(x, y, dx, dy)
dot.on_scroll        = fn(x_offset, y_offset)
```

**Example** — toggle color on click:
```python
self.color = dot.red

def mouse_pressed(x, y, b):
    self.color = dot.blue if self.color == dot.red else dot.red

dot.on_mouse_press = mouse_pressed
```

### Keyboard

```python
dot.on_key_press = fn(key, action, modifiers)
```

**Action types**
- `dot.keys.ACTION_PRESS`
- `dot.keys.ACTION_RELEASE`

**Key constants** — see [Key constants](#key-constants) in the appendix.

**Modifiers**
- `dot.modifiers.shift`
- `dot.modifiers.ctrl`
- `dot.modifiers.alt`

### Buttons

Create clickable UI buttons directly on the canvas.

```python
def setup(self):
    def on_click(btn):
        print(f"Button '{btn.text}' was clicked!")

    def on_hover(btn):
        print(f"Button '{btn.text}' was hovered!")

    dot.create_button(
        300, 250, 200, 50,
        text="Click Me",
        id="button1",
        on_release=on_click,
        on_hover=on_hover,
    )

def draw(self):
    dot.background((40, 40, 50))
    dot.update_buttons()
    dot.draw_buttons()
```

---

# Composition

## Transforms

Transforms apply to all subsequent drawing within their block. Always prefer the `with` form — it auto-restores state when the block ends.

### `with dot.transform():`

Save and scope the current transformation state.

```python
with dot.transform():
    dot.translate(400, 300)
    dot.rotate(0.5)
    dot.circle((0, 0), 50)  # rotated circle at (400, 300)
# transform state restored here
```

### `translate(x, y, z=0)`

Move the coordinate system.

```python
dot.translate(400, 300)
dot.circle((0, 0), 50)  # drawn at (400, 300)
```

### `rotate(angle, x=0, y=0, z=1)`

Rotate the coordinate system.

**Parameters**
- `angle` *(float)* — Rotation angle in **radians**.
- `x`, `y`, `z` — Axis of rotation (default is Z, which is what you want for 2D).

```python
import math

# 2D rotation (around Z)
dot.translate(400, 300)
dot.rotate(math.pi / 4)  # 45°
dot.rectangle((-50, -50), (50, 50))

# 3D rotation (around Y)
dot.rotate(math.pi / 6, x=0, y=1, z=0)
```

### `scale(s, y=None, z=None)`

Scale the coordinate system.

**Parameters**
- `s` *(float)* — X-axis scale factor (also used for Y and Z if they're omitted).
- `y`, `z` *(float)* — Per-axis scale factors.

```python
dot.scale(2.0)                     # uniform
dot.scale(2.0, 0.5)                # non-uniform (stretches)
```

### `reset_transforms()`

Reset all transformations to identity. Usually you want `with dot.transform():` instead — it auto-restores.

### Recipe: scale around a point

To scale around a specific point (e.g., the center of an image), translate → scale → translate back:

```python
center_x, center_y = 400, 300
with dot.transform():
    dot.translate(center_x, center_y, 0)
    dot.scale(2.0)
    dot.translate(-width/2, -height/2, 0)
    dot.paste(image, (0, 0))
```

## Layers

Layers are offscreen framebuffers for compositing. There's always an active render target — by default it's the main canvas, but inside a `with dot.layer(...)` block it's that layer instead.

Anything drawn to a layer is invisible until you explicitly `draw_layer()` it onto the canvas.

### `get_layer()`

Create a new layer.

**Returns:** *(int)* Layer ID.

```python
layer = dot.get_layer()
```

### `with dot.layer(layer_id):`

Render to a layer for the duration of the block.

```python
layer = dot.get_layer()
with dot.layer(layer):
    dot.circle((400, 300), 100)
    dot.rectangle((400, 300), (500, 500))
# drawing returns to the main canvas
```

### `draw_layer(layer_id, alpha=1.0)`

Draw a layer to the screen (or to another active layer).

**Parameters**
- `layer_id` *(int)* — Layer to draw.
- `alpha` *(float)* — Transparency (0.0 = invisible, 1.0 = opaque).

```python
dot.draw_layer(layer, alpha=0.5)
```

### `release_layer(layer_id)`

Free a layer's GPU resources.

```python
dot.release_layer(layer)
```

## Images

### `paste(image, position, size=None, alpha=1.0)`

Paste a NumPy array onto the canvas. Respects the current transform — use translate/scale for positioning and effects.

**Parameters**
- `image` *(np.ndarray)* — Image array. Supports:
  - `(H, W, 3)` — RGB
  - `(H, W, 4)` — RGBA
  - `(H, W)` — Grayscale
  - Values: `uint8` (0–255) or `float` (0.0–1.0).
- `position` *(Tuple[int, int])* — `(x, y)` top-left corner.
- `size` *(Tuple[int, int], optional)* — `(width, height)` to resize. `None` = original size.
- `alpha` *(float)* — Overall transparency.

See the `examples/video_and_images/` folder for image and webcam recipes (including OpenCV integration).

---

# Effects & shaders

Dorothy ships with a set of built-in video effects and also lets you write custom GLSL shaders. Both share the same `bake` parameter, explained [below](#baked-vs-display-filter).

## Built-in effects

All effects accept a `bake` parameter. Most also take effect-specific parameters. Effects are applied to the current canvas after your shapes are drawn.

### `pixelate(pixel_size=8.0, bake=False)`
Pixelate the canvas into larger blocks. Larger `pixel_size` = more pixelated.

```python
dot.circle((dot.mouse_x, dot.mouse_y), 50)
dot.pixelate(12.0)
```

### `blur(bake=False)`
Gaussian-style blur over the whole canvas.

```python
dot.circle((200, 200), 100)
dot.blur()
```

### `rgb_split(offset=0.01, bake=False)`
Chromatic-aberration / glitch effect. `offset` in the range 0.0–0.1.

```python
dot.circle((dot.mouse_x, dot.mouse_y), 50)
dot.rgb_split(0.02)
```

### `feedback(zoom=0.98, bake=True)`
Zoom-trail feedback. `zoom < 1.0` zooms out (trails grow outward); `> 1.0` zooms in. Defaults to `bake=True` because feedback only works when it compounds.

```python
dot.circle((dot.mouse_x, dot.mouse_y), 20)
dot.feedback(0.99)  # slow outward trails
```

### `roll(offset_x=0.0, offset_y=0.0, bake=True)`
Scroll/shift the canvas with wrap-around (like `np.roll`). Positive `offset_x` = right, positive `offset_y` = down.

```python
dot.circle((dot.width // 2, dot.height // 2), 50)
dot.roll(2.0, 0.0)  # continuously scrolls right
```

### `invert(bake=False)`
Invert all colors.

```python
dot.background(dot.white)
dot.fill(dot.red)
dot.circle((200, 200), 100)
dot.invert()
```

### `tile(grid_x=2, grid_y=2, bake=False)`
Repeat the canvas in an `grid_x × grid_y` grid.

```python
dot.circle((100, 100), 50)
dot.tile(4, 4)  # 4×4 grid of circles
```

### `cutout(color, threshold=0.1, bake=True)`
Make pixels of a specific color transparent (chroma key / green screen).

**Parameters**
- `color` — RGB color to remove, e.g. `(0, 0, 0)` or `dot.green`.
- `threshold` — Matching tolerance. 0.0 = exact match, 0.5 = loose.

```python
dot.background(dot.black)
dot.fill(dot.red)
dot.circle((200, 200), 100)
dot.cutout((0, 0, 0))  # remove black background
```

## Baked vs display filter

Every effect and shader has a `bake` flag. This is the single most important concept for compositing in Dorothy.

| Mode | Effect on canvas | Use for |
|---|---|---|
| `bake=True` | Writes result **back into** the canvas. Persists across frames (if you don't clear with `background()`). | Feedback, trails, decay, generative effects that compound |
| `bake=False` | Overlays the effect **on screen only**. Canvas stays unchanged. | Post-processing: blur, pixelate, color grade |

**Baked (feedback loop)**
```python
def draw(self):
    dot.circle((dot.mouse_x, dot.mouse_y), 20)
    dot.feedback(0.99, bake=True)  # leaves trails
# NOTE: no dot.background() — it would wipe the accumulated trails
```

**Display filter**
```python
def draw(self):
    dot.background(dot.white)
    dot.circle((200, 200), 50)
    dot.pixelate(8.0, bake=False)  # just a visual filter
```

### Combining effects

Chain multiple effects by calling them in sequence. Use `bake=True` on all but the last — each baked result passes through to the next effect:

```python
def draw(self):
    dot.circle((dot.mouse_x, dot.mouse_y), 30)
    dot.feedback(0.98, bake=True)     # zoom trails
    dot.rgb_split(0.015, bake=True)   # glitch
    dot.pixelate(6.0)                 # retro look (bake=False OK on the last one)
```

## Custom GLSL shaders

Apply arbitrary fragment shaders for effects beyond the built-ins.

### `apply_shader(shader_code, bake=True, **uniforms)`

**Parameters**
- `shader_code` *(str)* — GLSL fragment shader source.
- `bake` *(bool)* — See [Baked vs display filter](#baked-vs-display-filter).
- `**uniforms` — Any additional uniforms to pass to the shader (e.g. `time=1.5`, `amount=0.1`).

### Shader template

```glsl
#version 330

uniform sampler2D texture0;   // the canvas (always available)
uniform vec2 resolution;      // canvas size in pixels (optional)
uniform float my_parameter;   // any user-defined uniform you pass in

in vec2 v_texcoord;           // texture coords 0–1
out vec4 fragColor;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    // modify color here
    fragColor = color;
}
```

```python
dot.apply_shader(shader_code, bake=True, my_parameter=1.0)
```

### Example — feedback with fade (baked)

```python
feedback_shader = '''
#version 330
uniform sampler2D texture0;
uniform float fade;
in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    color.rgb *= fade;  // fade each frame
    fragColor = color;
}
'''

def draw():
    # Don't call background() — the canvas accumulates history
    dot.circle((dot.mouse_x, dot.mouse_y), 20)
    dot.apply_shader(feedback_shader, bake=True, fade=0.98)
```

### Example — blur (display filter)

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
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            color += texture(texture0, v_texcoord + vec2(x, y) * pixel);
        }
    }
    fragColor = color / 25.0;
}
'''

def draw():
    dot.background((0, 0, 0))  # safe: display-filter shader doesn't touch canvas
    dot.circle((dot.mouse_x, dot.mouse_y), 50)
    dot.apply_shader(blur_shader, bake=False)
```

### Shader tips

- **Baked shaders**: don't call `dot.background()` in your draw loop — it wipes the canvas each frame.
- **Display-filter shaders**: safe to call `dot.background()` — the shader is just an overlay.
- **Chaining**: call `apply_shader()` multiple times. Use `bake=True` on all but the last.
  ```python
  dot.apply_shader(self.pixelate, pixelSize=int(mean_amp * 100), bake=True)
  dot.apply_shader(self.rgb_split, bake=False, offset=mean_amp * 0.3)
  ```
- **Performance**: shaders with many texture lookups can drop framerate.
- **Debugging**: if a shader fails to compile, check the console for the GLSL error.

---

# Audio

Dorothy has a full audio system for playback, analysis, and generation. All audio runs on background threads so it won't block rendering.

**Audio section contents**
- [Audio sources](#audio-sources) — file, device (mic), DSP callback, samples, RAVE, MAGNet
- [Analysis](#audio-analysis) — amplitude, FFT, onset, beat detection
- [Signal smoothing](#signal-smoothing)
- [Playback control](#playback-control)
- [Sequencing](#audio-sequencing) — Clock, Sequence, Note
- [Instruments](#instruments) — PolySynth, Sampler, GranularSynth
- [Advanced](#advanced-audio-features) — RAVE timbre transfer, callbacks, gain
- [Performance tips](#audio-performance-tips)
- [Troubleshooting](#audio-troubleshooting)

All audio APIs live under `dot.music`. Running streams return an integer *output id* that you can use to index into `dot.music.audio_outputs[id]` for advanced control.

## Audio sources

Multiple sources can run simultaneously — each returns its own id.

### File playback

```python
file_id = dot.music.start_file_stream(
    "song.wav",
    fft_size=512,        # FFT window size for analysis
    buffer_size=1024,    # audio buffer size (larger = smoother, more latency)
    sr=44100,            # sample rate
    output_device=None,  # None = default output, or a device ID
    analyse=True,        # enable FFT and amplitude analysis
)
```

### Device input (microphone / system audio)

```python
device_id = dot.music.start_device_stream(
    device=2,            # device ID — see sd.query_devices()
    fft_size=1024,
    buffer_size=2048,
    sr=44100,
    analyse=True,
)
```

### Custom DSP (audio generation)

Pass an `audio_callback(size)` function that returns a NumPy array of `size` samples.

```python
import numpy as np
phase = 0

def audio_callback(size):
    global phase
    frequency = 440   # A4
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
    analyse=True,
)
```

### Pre-loaded sample buffer

```python
import numpy as np

# Generate or load audio into a NumPy array
samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)

sample_id = dot.music.start_sample_stream(
    samples,
    fft_size=1024,
    buffer_size=1024,
    sr=44100,
    output_device=None,
    analyse=True,
)
```

### RAVE neural vocoder

```python
rave_id = dot.music.start_rave_stream(
    model_path="vintage.ts",
    fft_size=1024,
    buffer_size=2048,
    sr=44100,
    latent_dim=16,       # must match the model
    output_device=None,
)

# Control the latent space
import torch
z = torch.randn(1, 16, 1)
dot.music.audio_outputs[rave_id].current_latent = z

# Or add a bias
dot.music.audio_outputs[rave_id].z_bias = torch.randn(1, 16, 1) * 0.1
```

### MAGNet spectral model

```python
magnet_id = dot.music.start_magnet_stream(
    model_path="model.pth",
    dataset_path="audio_samples.wav",
    buffer_size=2048,
    sr=44100,
    output_device=None,
)
```

### Listing audio devices

```python
import sounddevice as sd
print(sd.query_devices())   # list all audio devices
print(sd.default.device)    # default [input, output]
```

## Audio analysis

When a stream is started with `analyse=True`, these methods return real-time analysis. All accept `output=0` to pick which source to analyse (default = first).

### `amplitude(output=0, smooth=1)`

Current RMS amplitude (volume).

**Returns:** *(float)* 0.0 and up (can exceed 1.0 for loud signals).

```python
amp = dot.music.amplitude()
radius = 50 + amp * 200
dot.circle((400, 300), radius)
```

### `fft(output=0)`

Current FFT magnitude spectrum.

**Returns:** *(np.ndarray)* length `(fft_size // 2) + 1`.

```python
fft = dot.music.fft()
for i, magnitude in enumerate(fft[::4]):  # every 4th bin
    x = i * 20
    height = magnitude * 300
    dot.rectangle((x, 600), (x + 18, 600 - height))
```

### `is_onset(output=0)`

Detect if an onset (sudden energy spike) occurred since the last call. Works offline (on files) or online on device streams (less accurate online).

**Returns:** *(bool)*

```python
# Enable onset analysis on a device stream
o = dot.music.start_device_stream(1)
dot.music.audio_outputs[o].onset_detector.threshold = 0.5
dot.music.audio_outputs[o].analyse_onsets = True

# In draw()
if dot.music.is_onset():
    dot.fill((255, 0, 0))
```

### `is_beat(output=0)`

Detect if a beat occurred since the last call.

**Returns:** *(bool)*

```python
# Enable beat analysis
o = dot.music.start_device_stream(1)
dot.music.audio_outputs[o].onset_detector.threshold = 0.5
dot.music.audio_outputs[o].analyse_onsets = True
dot.music.audio_outputs[o].analyse_beats = True

# In draw()
if dot.music.is_beat():
    dot.fill((255, 0, 0))
    dot.circle((400, 300), 200)
```

> ⚠️ **Note:** Beat detection only works with `start_file_stream()` or `start_sample_stream()` — it requires real audio data and runs beat tracking at stream initialization.

## Signal smoothing

A utility for smoothing any noisy signal (audio analysis values, sensor data, etc.) with a rolling window.

### `get_window(window_size, method="average", dims=1)`

Create a smoothing window.

**Parameters**
- `window_size` *(int)* — Number of values to average over.
- `method` *(str)* — Smoothing method.
- `dims` *(int)* — Dimensionality of inputs. Use `1` for scalars, higher for vectors.

### `.add(value)`

Add a new value; returns the current smoothed value.

**Examples**
```python
# 1D
w = dot.get_window(10)
mean = w.add(new_val)

# Multi-dimensional
w = dot.get_window(10, dims=3)
mean = w.add([x, y, z])
```

## Playback control

All playback functions take an optional `output=0` parameter to target a specific stream.

```python
dot.music.play()        # start / resume
dot.music.play(1)       # second output

dot.music.stop()        # stop completely
dot.music.pause()       # pause (resumable)
dot.music.resume()      # resume from pause
```

## Audio sequencing

`Note`, `Sequence`, `Clock`, and the instruments ([`PolySynth`](#polysynth), [`Sampler`](#sampler), [`GranularSynth`](#granularsynth)) form a unified sequencing system. A `Sequence` is connected to a `Clock` and an instrument — the clock drives the sequence, and the sequence fires `note_on` / `note_off` on the instrument.

**Minimal example**
```python
from dorothy.Audio import Sequence, Note

def setup(self):
    self.clock = dot.music.get_clock(bpm=120)
    self.clock.set_tpb(4)                        # 4 ticks per beat

    idx = dot.music.start_poly_synth_stream()
    self.synth = dot.music.audio_outputs[idx]

    self.seq = Sequence(steps=8, ticks_per_step=4)
    self.seq[0] = Note(60)                        # middle C on step 0
    self.seq.connect(self.clock, self.synth)
    self.clock.play()
```

### Note

A dataclass representing a single step event.

```python
Note(
    midi,             # MIDI note number (0–127). Middle C = 60, A4 = 69
    vel=0.8,          # velocity 0.0–1.0
    duration=1,       # duration in steps before note_off fires
    # Per-note ADSR overrides (None = use instrument default)
    attack=None,
    decay=None,
    sustain=None,
    release=None,
    # Per-note oscillator overrides (PolySynth only)
    waveform=None,    # 'sine'|'saw'|'triangle'|'noise'|'supersaw'|'fm'|'pwm'
    fm_ratio=None,
    fm_index=None,
    detune=None,
    n_oscs=None,
    pwm=None,
)

note.freq   # read-only: MIDI → Hz  (440 * 2**((midi - 69) / 12))
```

### Clock

Tempo-synced timing. Runs on a background thread.

**Create**
```python
self.clock = dot.music.get_clock(bpm=120)
self.clock.set_tpb(4)   # ticks per beat (default 4)
```

**Properties**
```python
self.clock.tick_ctr        # current tick count (increments before callbacks fire)
self.clock.bpm             # current BPM
self.clock.ticks_per_beat  # subdivisions per beat
self.clock.playing         # True if running
self.clock.tick_length     # milliseconds per tick
```

**Methods**
```python
self.clock.play()             # start (resets tick_ctr to 0)
self.clock.stop()             # stop
self.clock.set_bpm(120)       # change tempo
self.clock.set_tpb(4)         # change tick subdivision

# Register extra callbacks (multiple supported)
self.clock.on_tick_fns.append(self.my_fn)
```

**Timing grid**
```python
# 4/4 — 16th-note steps
self.clock.set_tpb(4)   # 4 ticks/beat  → ticks_per_step=1 → 16ths
                        #                → ticks_per_step=4 → quarters
# Tip: call set_tpb() after set_bpm() so tick_length recalculates
```

### Sequence

Step sequencer that drives any compatible instrument.

**Create and connect**
```python
seq = Sequence(steps=16, ticks_per_step=1)
seq.connect(clock, synth)   # registers tick callback; call before clock.play()
```

**Step editing**
```python
seq[i] = Note(60)               # single note
seq[i] = [Note(60), Note(64)]   # chord
seq[i] = []                     # rest
note = seq[i]                   # read a step

seq.steps = 32                  # resize (current_step wraps into new range)
seq.ticks_per_step = 2          # change step resolution live
```

**Pattern methods**
```python
seq.clear()           # empty all steps
seq.clear(i)          # empty one step
seq.set_pattern([     # replace all steps atomically; fires all_notes_off first
    [Note(60)],
    [],
    [Note(64), Note(67)],
    [],
])
seq.all_notes_off()   # immediately release all pending notes
```

## Instruments

All instruments are `AudioDevice`s that respond to `note_on` / `note_off`. They can be driven by a [`Sequence`](#sequence) or called directly (thread-safe).

### PolySynth

Polyphonic synthesizer. Up to `n_voices` simultaneous notes.

**Create**
```python
idx = dot.music.start_poly_synth_stream(
    n_voices=8,
    n_harmonics=4,
    attack=0.01, decay=0.1, sustain=0.7, release=0.3,
    waveform='sine',   # default oscillator shape
    buffer_size=512,
    sr=44100,
)
synth = dot.music.audio_outputs[idx]
```

**Waveforms**: `'sine'` · `'saw'` · `'triangle'` · `'noise'` · `'supersaw'` · `'fm'` · `'pwm'`.

**Default parameters** (read/write)
```python
synth.attack        # ADSR attack (s)
synth.decay         # ADSR decay  (s)
synth.sustain       # ADSR sustain level 0–1
synth.release       # ADSR release (s)
synth.waveform      # default oscillator shape
synth.fm_ratio      # FM: modulator = fm_ratio × carrier  (default 2.0)
synth.fm_index      # FM: modulation depth in radians     (default 1.0)
synth.detune        # supersaw: total semitone spread     (default 0.2)
synth.n_oscs        # supersaw: oscillator count          (default 7)
synth.pwm           # PWM: duty cycle 0–1 (0.5 = square)
```

**Direct API** (thread-safe)
```python
synth.note_on(freq, vel=0.8, waveform='saw', attack=0.05, ...)
synth.note_off(freq)
synth.all_notes_off()
```

Per-note overrides in `note_on` apply to that note only; `None` falls back to the synth default. Notes passed through a `Sequence` carry overrides from their `Note` fields.

### Sampler

Sample player. `Note.midi` is used as the **slot index**, `Note.vel` scales volume.

**Create**
```python
idx = dot.music.start_sampler_stream(
    paths=["kick.wav", "snare.wav", "hat.wav"],   # optional pre-load
    sr=44100,
    buffer_size=512,
)
sampler = dot.music.audio_outputs[idx]
sampler.load(["kick.wav", "snare.wav"])   # load or swap at any time
```

Slot 0 = `paths[0]`, slot 1 = `paths[1]`, etc.

**Sequence usage**
```python
seq[0] = Note(0, vel=1.0)   # trigger slot 0
seq[2] = Note(1, vel=0.8)   # trigger slot 1
seq.connect(clock, sampler)
```

Samples play to their natural end; `note_off` is a no-op.

**Direct API**
```python
sampler.trigger(0, vel=1.0)    # trigger by slot index
sampler.all_notes_off()         # stop all voices immediately
```

### GranularSynth

Loads a single audio file and plays it as overlapping short grains.

`Note.midi` 69 (A4, 440 Hz) = original file pitch. Other values shift pitch by semitone distance from A4. `Note.vel` scales voice volume.

**Create**
```python
idx = dot.music.start_granular_stream(
    path="texture.wav",     # optional pre-load
    position=0.5,           # initial read position (0–1)
    spread=0.1,
    grain_size=80.0,        # ms
    density=8.0,            # grains/sec/voice
    attack=0.3, decay=0.3,
    n_grains=32,
    pitch=0.0,              # semitones
    pitch_spread=0.0,       # per-grain jitter (semitones std dev)
    sr=44100,
    buffer_size=512,
)
gran = dot.music.audio_outputs[idx]
gran.load("texture.wav")    # load or swap source at any time
```

**Parameters** (read/write at any time)
```python
gran.position      # 0–1, read-head centre in source
gran.spread        # 0–1, random position scatter
gran.grain_size    # ms per grain
gran.density       # grains per second per voice
gran.attack        # fraction of grain for fade-in
gran.decay         # fraction of grain for fade-out
gran.n_grains      # max simultaneous grains
gran.pitch         # global semitone shift
gran.pitch_spread  # per-grain pitch jitter (std dev)
```

**Direct API**
```python
gran.note_on(freq, vel=0.8)   # start grain cloud at pitch/volume
gran.note_off(freq)            # stop spawning; active grains play out
gran.all_notes_off()           # silence immediately
```

## Advanced audio features

### RAVE timbre transfer

Route one audio source through a RAVE model to re-synthesize its timbre:

```python
rave_id = dot.music.start_rave_stream("vintage.ts")
mic_id  = dot.music.start_device_stream(device=0)
dot.music.update_rave_from_stream(mic_id)
# RAVE now encodes the mic input and generates audio
```

### Custom callbacks

Access raw audio buffers as they arrive:

```python
def on_new_frame(buffer):
    print(f"New audio: {buffer.shape}, max: {np.max(np.abs(buffer))}")

file_id = dot.music.start_file_stream("song.wav")
dot.music.audio_outputs[file_id].on_new_frame = on_new_frame
```

### Gain control

Adjust volume per source:

```python
file_id = dot.music.start_file_stream("song.wav")
dot.music.audio_outputs[file_id].gain = 0.5   # 50% volume
dot.music.audio_outputs[file_id].gain = 0.0   # mute
```

### Multiple audio outputs

You can run many sources at once. See `examples/audio_playback/multi_audio_outputs.py`.

## Audio performance tips

**Buffer size** — larger = smoother audio, more latency.
- Live input: 512–1024
- Playback: 1024–2048
- Glitching? try 4096.

**FFT size** — balance frequency vs time resolution.
- Music: 1024–2048
- Speech: 512–1024
- Real-time responsiveness: 512.

**Cache analysis values** — read `fft()` / `amplitude()` **once per frame**, not repeatedly:
```python
def draw(self):
    # GOOD
    fft = dot.music.fft()
    amp = dot.music.amplitude()
    for i, val in enumerate(fft):
        ...

    # BAD — recomputes each call
    # for i in range(len(dot.music.fft())):
    #     val = dot.music.fft()[i]
```

**Reduce visual complexity** — subsample data when rendering:
```python
fft = dot.music.fft()[::4]   # every 4th value
```

## Audio troubleshooting

### Glitches / crackling

**Cause:** draw loop blocking the audio thread.
**Fix:**
- Increase `buffer_size` (try 4096).
- Simplify `draw()`.
- Reduce FFT size.
- Keep VSync on (default in ModernGL build).

### No audio output

```python
import sounddevice as sd
print(sd.query_devices())  # list devices
print(sd.default.device)   # check default
```

Set a device explicitly:
```python
dot.music.start_file_stream("song.wav", output_device=2)
```

### FFT values all zero

- `analyse=False` was set — enable it.
- Buffer size too small.
- No audio actually playing.

```python
dot.music.start_file_stream("song.wav", analyse=True, buffer_size=2048)
dot.music.play()
```

### Beat detection not working

- Only works with `start_file_stream()` or `start_sample_stream()` — not live input.
- Beat tracking runs once at stream initialization; needs a real audio file.

---

# Live coding

Dorothy can reload your sketch every time you save the file.

**Three changes from the standard template:**

1. **No `__init__()`** in the `MySketch` class.
2. **Don't instantiate** `MySketch` yourself.
3. **Call `start_livecode_loop()`** at the bottom of the file.

```python
from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    def setup(self):
        self.col = (0, 255, 0)
        print("start")

    def run_once(self):
        print("run once")
        self.col = (0, 0, 0)

    def draw(self):
        dot.background(self.col)
        dot.fill(dot.blue)
        dot.rectangle((0, dot.frames % 40), (400, 100))

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
```

Now edit the file and save — changes appear instantly.

## The `run_once()` method

A special method that runs **once** every time the file is reloaded. Use it to reset state when you change your code:

```python
def run_once(self):
    # Reset state on every code update
    self.particles = []
    self.angle = 0
    print("Code reloaded, state reset!")
```

---

# Appendix

## Color constants

Predefined colors — all standard CSS colors are available as `dot.<color_name>`.

```python
dot.fill(dot.red)
dot.circle((400, 300), 50)

dot.fill(dot.cornflowerblue)
dot.fill(dot.hotpink)
```

## Key constants

For use with `dot.on_key_press` — see [Interaction callbacks](#interaction-callbacks).

**Common keys**
- `dot.keys.SPACE`
- `dot.keys.ENTER`
- `dot.keys.ESCAPE`
- `dot.keys.TAB`
- `dot.keys.BACKSPACE`
- Letters: `dot.keys.A` … `dot.keys.Z`
- Numbers: `dot.keys.NUMBER_0` … `dot.keys.NUMBER_9`
- Arrows: `dot.keys.UP`, `dot.keys.DOWN`, `dot.keys.LEFT`, `dot.keys.RIGHT`

**Actions**
- `dot.keys.ACTION_PRESS`
- `dot.keys.ACTION_RELEASE`

**Modifiers**
- `dot.modifiers.shift`
- `dot.modifiers.ctrl`
- `dot.modifiers.alt`

## Troubleshooting

### Audio glitches
See [Audio troubleshooting](#audio-troubleshooting).

### Shapes not visible
- Check camera mode: `dot.camera_2d()` for flat drawing.
- Check fill/stroke: make sure a color is set.
- Check coordinates: are they inside the window?

### Transforms not working
- Use `with dot.transform():` to scope changes to a block.
- Make sure your drawing code is **inside** the block.
- Apply transforms **before** drawing.
- Remember transforms accumulate within a block.

### Images upside down
- Dorothy handles orientation automatically.
- If an image still looks flipped: `img = np.flipud(img)`.

### Mouse not responding
- Mouse position is polled every frame — `dot.mouse_x` and `dot.mouse_y` update automatically.
- If you need to *react* to clicks, use the callbacks under [Interaction callbacks](#interaction-callbacks).

## Tips & best practices

### Performance
- **Minimize state changes** — batch drawing with the same fill/stroke.
- **Use layers for static content** — draw once, reuse many times.
- **Cache audio analysis** — call `fft()` / `amplitude()` once per frame, not per item.
- **Subsample data** when rendering lots of points.

### Transforms
- **Order matters** — the typical order is translate → rotate → scale.
- **Scale around a point** — translate to point, scale, translate back. See [the recipe](#recipe-scale-around-a-point).

### Debugging
- **`annotate=True`** on shape calls — shows coordinates.
- **Periodic prints** — `if dot.frames % 60 == 0: print(...)`.
- **Mouse position** — `print(dot.mouse_x, dot.mouse_y)`.

### Code organisation
- **Use a class** for state — store variables on `self`.
- **Do expensive setup once** in `setup()`.
- **Keep `draw()` fast** — it runs ~60 times per second.

## Version history

### ModernGL refactor (current)
- GPU-accelerated rendering with ModernGL.
- Native 3D support.
- 10–100× performance improvement.
- Backward-compatible API with the original version.
- Transform-aware image pasting.
- Layer system with alpha blending.

## Credits

Dorothy by Louis McCallum.
ModernGL refactor maintains API compatibility while adding GPU acceleration and 3D support.
