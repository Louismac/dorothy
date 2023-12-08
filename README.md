# Dorothy
A Creative Computing Python Library for Audio Reactive Drawing

Leaning on the work of ``openCV``, ``sounddevice`` and ``librosa`` with a ``Processing / p5.js`` -like API, make some easy sketching with shapes and images reacting to FFT, Beats and Amplitude in Python!

<img src="examples/images/output2.gif" alt="drawing" width="200"/><img src="examples/images/output3.gif" alt="drawing" width="200"/><img src="examples/images/output4.gif" alt="drawing" width="200"/>

## Dorothy.py

Has ``setup()`` and ``draw()`` functions that can be overwritten using a custom class

```
from Dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        dot.background((255,255,255))
    
    def draw(self):
        return

MySketch()   
```

### Running

Launch your sketch (`.py`) from terminal. Your sketch **must** be in the same directory as ``Dorothy.py``

### Closing the window

Either hold `q` with the window in focus, or use `ctrl-z` in the terminal to close. You must close the current window before re-running to see changes.

## Drawing

For drawing, its suggested to use the [openCV drawing functions](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html), with ``dot.canvas`` as the first argument (this is the image to draw onto). 

``dot.canvas`` is just a 3 channel 4D numpy array that you can edit you like in the sketch or draw functions

Other Processing like functions are 

* `dot.background((colour), alpha)`

* `dot.mouse_x`, `dot.mouse_y`, ``dot.mouse_down``

* ``dot.millis``, ``dot.frames``

* ``dot.width``, ``dot.height``


## Reacting to Music

Makes analysis information available for real time visualisation. Works out the threading to work with ``Dorothy.py``

We have 

* ``dot.music.fft_vals`` (updates with music playback)

* ``dot.music.amplitude`` (updates with music playback)

* ``dot.music.is_beat()`` (call in ``draw()``, returns true if there has been a beat since last frame

You can either play a soundfile 

```
file_path = "../audio/hiphop.wav"
dot.music.load_file(file_path, fft_size = 1024, buffer_size = 1024)
```

Or pick a an output device playing on your computer. On MacOSX I use [Blackhole](https://existential.audio/blackhole/download/) and [Multioutput device](https://support.apple.com/en-gb/guide/audio-midi-setup/ams7c093f372/mac) to pump audio to here, and to listen in speakers as well. Should work on windows but I havent tested anything yet!

```
print(sd.query_devices())
dot.music.get_stream(2, fft_size=1024, buffer_size=2048)
```

Both use 

```
dot.music.play()
dot.music.stop()
dot.music.pause()
dot.music.resume()
```

## Examples

### [Seed](examples/seed.py) 

This is the bare bones starter for any projects 

### [Alpha](examples/alpha.py) 

Shows how to draw shapes with transparency. openCV doesnt do this natively so you call ``dot.to_alpha(alpha_val)`` to get a new layer to draw to (instead of ``dot.canvas``). We then take care of the masking and blending to make it work. 

### [Alpha Background](examples/trails.py) 

Add a second argument to the `an.background()` to draw a transparent background and make trails

### [Grid](examples/grid.py) 

Audio reactive color pattern from a nested for loop in the ``def draw()`` function 

### [Molnar](examples/molnar.py) 

Audio reactive scaled pattern from a nested for loop in the ``def setup()`` function 

### [Mouse Position](examples/mouse.py) 

Use ``dot.mouse_x`` and ``dot.mouse_y`` to control where a circle is drawn, with size moving to amplitude.

### [Webcam Capture](examples/video.py) 

Use openCV to grab and draw the webcam and bounce centre panel to music.

### Linear Transforms

Apply linear transforms and translations to canvases. This works in the opposite way to Processing, in that you 

1. Get a new canvas (``dot.push_layer()``)

2. Draw to it 

3. Apply transformations (``dot.transform()``,``dot.rotate()``,``dot.scale()``). This function also takes a new origin about which to make the transformation if required (a translation).

4. Put back onto main canvas (``dot.pop_layer()``)

[Rotate](examples/rotate.py)

[Rotate In a Grid](examples/rotate_grid.py)

[Scale](examples/scale.py)
 

### [Beat Tracking](examples/beats.py) 

Shows how to use ``dot.music.is_beat()`` to display the beat tracking data in real time. Also shows how to use properties of the ``MySketch`` class to have variables that persist outside of the ``def draw()`` and ``def setup()`` functions.

### [FFT](examples/fft.py) 

Visualise live fft data

### [Amplitude](examples/amplitude.py) 

Visualise live amplitude data

### [Images](examples/many_images.py)

Use `get_images()` to load in a dataset of images and `dot.paste()` to copy onto canvas

### [Contours](examples/contours.py)

Get contours and mask out, moving image sections radially in response to fft values. More complex example!








