# Dorothy
A Creative Computing Python Library for Interactive Audio Generation and Audio Reactive Drawing. Now at v.0.0.5!

Leaning on the work of ``openCV``, ``sounddevice`` and ``librosa`` with a ``Processing / p5.js`` -like API, make some easy sketching with shapes and images reacting to FFT, Beats and Amplitude in Python! Also, as its Python and the canvas is just a `numpy` pixel array, you can do any of the cool Python stuff you would normally do, or use other powerful libraries like `NumPy`, `PyTorch` or `Tensorflow`.

<img src="images/output2.gif" alt="drawing" width="200"/><img src="images/output3.gif" alt="drawing" width="200"/><img src="images/output4.gif" alt="drawing" width="200"/>

* [Brief Guide](https://github.com/Louismac/dorothy/blob/main/README.md#brief-guide)

* [Examples](https://github.com/Louismac/dorothy/blob/main/README.md#examples)

* [Full API Reference](https://github.com/Louismac/dorothy/blob/main/Reference.md)

* [Generating Music](https://github.com/Louismac/dorothy/blob/main/README.md#music-generation)


##  Installation

```pip3 install dorothy-cci```

##  Brief Guide

Has ``setup()`` and ``draw()`` functions that can be overwritten using a custom class

```python
from dorothy import Dorothy

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

1. ``pip install dorothy-cci``

2. Try out one of the examples

### Closing the window

Either hold `q` with the window in focus, or use `ctrl-z` in the terminal to close. You must close the current window before re-running to see changes.

### Drawing

For drawing, we have wrappers around the [openCV drawing functions](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html) for `circle`, `line`, `rectangle` and `poly`.

We use the well established `fill` and `stroke` approach for colouring and borders. You can also annotate the coordinates of your shape in place for debugging!

```python
def draw(self):
    dot.fill(dot.black)
    dot.stroke(dot.red)
    dot.set_stroke_weight(5)
    dot.circle((200,200), 100, annotate = True)
```

Everything is drawn to ``dot.canvas``, which is just a 3 channel 4D numpy array that you can edit you like in the sketch or draw functions

Other Processing like functions are 

* `dot.background((colour), alpha)`

* `dot.mouse_x`, `dot.mouse_y`, ``dot.mouse_down``

* ``dot.millis``, ``dot.frames``

* ``dot.width``, ``dot.height``

### Live Coding / Realtime updates

You can update ``Dorothy`` sketches on the fly and see results without re-running code using the [example](examples/livecode/). Here we split the code into 

1. [livecode.py](examples/livecode/livecode.py) that is launched a left alone. 

2. [sketch.py](examples/livecode/sketch.py) that has you `MySketch` class in. You can make changes to this and when you save the file, the changes will be reflected in the `Dorothy` window.

### Reacting to Music

Makes analysis information available for real time visualisation. Works out the threading to work with ``Dorothy.py``

We have 

* ``dot.music.fft_vals`` (updates with music playback)

* ``dot.music.amplitude`` (updates with music playback)

* ``dot.music.is_beat()`` (call in ``draw()``, returns true if there has been a beat since last frame


## Examples

### [Seed](examples/seed.py) 

This is the bare bones starter for any projects 

### [Alpha](examples/alpha.py) 

Shows how to draw shapes with transparency. openCV doesnt do this natively so you call ``dot.get_layer()`` to get a new layer to draw to (instead of ``dot.canvas``). We then take care of the masking and blending to make it work when you do `dot.draw_layer(layer, alpha_val)`. 

### [Alpha Background](examples/trails.py) 

Use an alpha layer and a translucent rectangle (instead of a opaque background) and make trails

### [Grid](examples/grid.py) 

Audio reactive color pattern from a nested for loop in the ``def draw()`` function 

### [Molnar](examples/molnar.py) 

Audio reactive scaled pattern from a nested for loop in the ``def setup()`` function 

### [Mouse Position](examples/mouse.py) 

Use ``dot.mouse_x`` and ``dot.mouse_y`` to control where a circle is drawn, with size moving to amplitude.

### [Webcam Capture](examples/video.py) 

Use openCV to grab and draw the webcam and bounce centre panel to music.

### Linear Transforms for Shapes

You can apply linear transforms and translations to for the built in shapes. This is more efficient that doing it the whole layer (see below). 

This works similarly to processing in that 

1. Apply transform with optional origin (``dot.translate()``,``dot.rotate()``,``dot.scale()``).

2. Draw shapes (`dot.circle()`,`dot.rectangle()`,`dot.line()`,`dot.poly()`). All transformations set previously will be applied.

3. Optionally `reset_transforms()` within the draw loop to isolate changes. 

4. Transforms are reset at the end of the draw loop.

[Rotate](examples/rotate.py)

[Rotate In Grid](examples/rotate_grid.py)

[Scale](examples/scale.py)

### Linear Transforms for Layers

You can apply linear transforms and translations to whole layers. This is useful if your drawing contains non-primitive shapes or images, or if for efficiency you draw something complex to a layer once then just redraw and transform that layer. 

It works in the opposite way to Processing, in that you 

1. Get a new canvas (``dot.get_layer()``)

2. Draw to it 

3. Apply transformations (``dot.transform_layer()``,``dot.rotate_layer()``,``dot.scale_layer()``). This function also takes a new origin about which to make the transformation if required (a translation).

4. Put back onto main canvas (``dot.draw_layer()``)

[Rotate Layer](examples/rotate_layer.py)

[Scale Layer](examples/scale_layer.py)
 

### [Beat Tracking](examples/beats.py) 

Shows how to use ``dot.music.is_beat()`` to display the beat tracking data in real time. Also shows how to use properties of the ``MySketch`` class to have variables that persist outside of the ``def draw()`` and ``def setup()`` functions.

### [FFT](examples/fft.py) 

Visualise live fft data

### [Amplitude](examples/amplitude.py) 

Visualise live amplitude data

### [Images](examples/many_images.py)

Use `get_images()` to load in a dataset of images and `dot.paste()` to copy onto canvas

### [Alpha Images](examples/load_image_alpha.py)

Load `.png` files with transparency

### [Hand Tracking](examples/hands.py)

With media pipe.

### [Face Tracking](examples/face-filter-pixels.py)

Track and pixelate face using Haar Cascades.

### [Contours](examples/contours.py)

Get contours and mask out, moving image sections radially in response to fft values. More complex example!

### [RAVE](examples/rave.py)

Examples on generating with / interacting with RAVE models

### [MAGNet](examples/magnet.py)

Examples on generating with / interacting with MAGNet models

### [YOLO Body Tracking](examples/yolo.py)

Example drawing pose from web cam

### [Interactive GAN](examples/gan.py)

Use the mouse to move through the latent space of an MNIST GAN

### [Interactive RAVE / YOLO](examples/yolo_rave.py)

Control interpolation points of RAVE with pose tracked hand position 

## Music Generation

### Picking Music Source

You can either play a soundfile 

```
file_path = "../audio/hiphop.wav"
dot.music.start_file_stream(file_path)
```

Or pick a an output device playing on your computer. On MacOSX I use [Blackhole](https://existential.audio/blackhole/download/) and [Multioutput device](https://support.apple.com/en-gb/guide/audio-midi-setup/ams7c093f372/mac) to pump audio to here, and to listen in speakers as well. Should work on windows but I havent tested anything yet!

You could also use this approach to get in the stream of your laptops microphone, or an external microphone. `print(sd.query_devices())` will give the you list of available devices, and their device ids to pass to the set up function.

```
print(sd.query_devices())
dot.music.start_device_stream(2)
```

Both use 

```
dot.music.play()
dot.music.stop()
dot.music.pause()
dot.music.resume()
```

### Custom DSP Loop

[Simple Synth](examples/synth.py)

If you want to make your own samples (e.g. from a synth, or from your own music geneation model), you just need to make a `dot.music.start_dsp_stream()` and pass it an audio callback where you return the samples as requested.

```python
def get_frame(size):
    #Get parameters from mouse
    frequency = dot.mouse_x
    amplitude = dot.mouse_y/dot.height
    #Get increments
    delta = 2 * np.pi * frequency / sr 
    x = delta * np.arange(size)
    #Make some sound
    audio = amplitude * np.sin(self.phase + x)
    #update phase
    self.phase += delta * size 
    return audio

dot.music.start_dsp_stream(get_frame, sr = sr)
```

### Generating Audio with [RAVE](https://github.com/acids-ircam/RAVE)

There is also a player to generate, visualise and interact with pretrained RAVE models. 

[Examples here](examples/rave.py)

```python
rave_id = dot.music.start_rave_stream("vintage.ts", latent_dim=latent_dim)
```

Will load in a `.ts` model. Remember to `play()` to start!

It will initially just start at a random place in latent space but there are two key ways to interact 

1. Manually set the z vector using  where z is a torch tensor and has the shape (1, latent_dims, 1)

```python
z = torch.randn((1,16,1))
dot.music.update_rave_latent(z)
```

3. Do timbre transfer from audio

   * Pipe audio from a stream you have already started (e.g. blackhole to pull whatever is coming from your computer, or a microphone). If you want to listen to the output of the RAVE model, you should manually set its output device so that it doesnt interfere with the stream you have hi-jacked from your machine.

```python
# Give an output device (e.g. your speakers) so you can hear the output
rave_id = dot.music.start_rave_stream("vintage.ts", output_device=4, latent_dim=latent_dim)
# Set stream to be blackhole / microphone device
device_id = dot.music.start_device_stream(2)
dot.music.update_rave_from_stream(device_id)
dot.music.play()
```
  
   * Or a file player stream

```python
rave_id = dot.music.start_rave_stream("vintage.ts", latent_dim=latent_dim)
device_id = dot.music.start_file_stream("../audio/gospel.wav")
# Set as input to rave (this mutes the source stream, use .gain property to hear both)
dot.music.update_rave_from_stream(device_id)
dot.music.play()
```
  
#### Z Bias 

You can also add a constant bias to the z vector to allow for some controllable / random variation. 

If you want to change this over time, you can use the `on_new_frame` callback. This is called whenever the chosen audio device (in this case the RAVE audio player) requests a new buffer and this function returns that buffer (so you can get the size, or do any custom analysis)

##### New random bias every frame
```python
def on_new_frame(buffer=np.zeros(2048)):
    n= len(buffer)
    #Update a new random 
    dot.music.audio_outputs[0].z_bias = torch.randn(1,latent_dim,1)*0.05

dot.music.audio_outputs[rave_id].on_new_frame = on_new_frame
```

##### Oscillating bias at a given frequency 
```python
def sine_bias(frame_number, frequency=1, amplitude=1.0, phase=0, sample_rate=44100):
    t = frame_number / sample_rate
    value = amplitude * math.sin(2 * math.pi * frequency * t + phase)
    return value

self.ptr = 0
def on_new_frame(buffer=np.zeros(2048)):
    n= len(buffer)
    #update with oscilating bias
    val = sine_bias(self.ptr, 5, 0.4)
    dot.music.audio_outputs[0].z_bias = torch.tensor([val for n in range(latent_dim)]).reshape((1,latent_dim,1))
    self.ptr += n

dot.music.audio_outputs[rave_id] = on_new_frame
```

### Generating Audio with [MAGNet](https://github.com/Louismac/MAGNet)

MAGNet is a lightweight LSTM spectral model. You can train models [here](https://github.com/Louismac/MAGNet) with as little as 30 seconds of audio in minutes. 

This generates in realtime given a trained model the original source audio file / dataset (to use as an impulse)

```python
dot.music.start_magnet_stream("models/magnet_wiley.pth", "../audio/Wiley.wav")
```










