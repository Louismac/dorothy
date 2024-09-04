
## Class `Dorothy`

This is the main class of the Library

### Properties:
- `width`
  - `int` representing canvases width
- `height`
  - `int` representing canvases height
- `layers`
  -  `list` with the current layers on. Layers are rendered sequentially from 0 (e.g. the last layer is at the front)
- `canvas`
  -  `np.array` of the current output canvas (what you see on the screen). 3 channels (RGB). 
- `recording`
  - `bool` determining whether screen recording is happening or not
- `mouse_x`
  - `int` giving mouse x corrdinate on screen. 0 is left most edge.
- `mouse_y`
  - `int` giving mouse y corrdinate on screen. 0 is top most edge.
- mouse_down
  - 'bool' is the mouse is currently pressed. 
- `frame`
  - `int` giving frame counter since program launched
- `millis`
  - `int` giving milliseconds since program launched
- `music`
  - an instance of `Audio` that controls the music analysis and generation

### Methods:
- `push_layer()`
  - Returns a new layer (`np.array') to draw to.
- `pop_layer(c)`
  - Push layer `c` onto the layer stack to be drawn
- `to_alpha(alpha=1)`
  - Get a new layer for transparency drawing. `alpha' is a float from 0 to 1, with 0 being fully transparent, 1 fully opaque 
- `linear_transformation(self, src, a, origin=(0,0)`
  - Perform a linear transformation to layer `src` given matrix `a` about a given `origin`. Returns transformed layer.
- `scale(canvas, sx=1, sy=1, origin=(0,0)`
  - Scale `canvas` given `sx` and `sy` factors and an `origin`. Returns transformed layer.
- `rotate(canvas, theta, origin=(0,0))`
  - Rotate `canvas` `theta` radians about `origin`. Returns transformed layer.
- `background(col, alpha=None)`
  - Set the background `col` (an RGB tuple). The overwrites all other layers. 
- `paste(layer, to_paste, coords=(0,0))`
  - paste a given image (`np.array` of pixels) onto a given `layer` at `coords` (given for the top left). Returns the `layer` after paste operation
- `update_canvas()`
  - Render the `layers` to the `canvas` 

## Class `Audio`

Controls the music analysis and generation. Is accessed through the `dot.music` property.

### Properties:
- `audio_outputs`
  - `list` of audio output devices (`AudioDevice`s). 
- `fft_vals`
  - `np.array` of most recent fft values from current audio output. Size is determined by `fft_size`.
- `amplitude`
  - `float` giving current mean amplitude.
- `tempo`
  - `float` of the tempo track (if a `FilePlayer` is being used (not a stream))


### Methods:
- `start_magnet_stream(model_path, dataset_path, buffer_size=2048, sr=44100, output_device=None)`
  - Given a `model_path` to a trained MAGNet model and a `dataset_path` to an seed audio `.wav` file, make a `MAGNetPlayer`. `output_device` is the audio device to output the generated audio to. A list of ids can be  found by executing `print(sd.query_devices())`. Returns index to device on the `audio_outputs` list.
- `start_rave_stream(model_path="",fft_size=1024, buffer_size=2048, sr = 44100, latent_dim=16, output_device=None)`
  - Given a `model_path` to a trained RAVE model, make a `RAVEPlayer`. Returns index to device on the `audio_outputs` list. `output_device` is the audio device to output the generated audio to. A list of ids can be  found by executing `print(sd.query_devices())`. Returns index to device on the `audio_outputs` list.
- `start_device_stream(device, fft_size=1024, buffer_size=2048, sr = 44100, output_device=None, analyse=True)`
  - Capture audio from an input (e.g. a microphone), passing the `device` id.  If using `output_device`, be careful of feedback!. A list of ids can be  found by executing `print(sd.query_devices())`. Returns index to device on the `audio_outputs` list.
- `start_file_stream(file_path, fft_size=1024, buffer_size=2048, sr = 44100, output_device=None, analyse = True)`
  - Start playback of a `.wav` file at `file_path`. `output_device` is the audio device to output the audio to, although this can be left empty if using the stream as input to another device (e.g a `RAVEPlayer`). A list of ids can be  found by executing `print(sd.query_devices())`. Returns index to device on the `audio_outputs` list.
- `update_rave_from_stream(input=0)`
  - Select the input stream to drive RAVE with if doing timbre transfer.  A list of ids can be  found by executing `print(sd.query_devices())`.
- `play()` 
  - Starts all `audio_outputs`.
- `stop()`
  - Stops all `audio_outputs`
- `is_beat()`
  - Return `True` if there as a beat since the last time this function was called. Should be called from within the `draw()` loop on every frame for best results.     

## Class `AudioDevice`

This is the parent class for all audio devices / players. Not actually instantiated.

### Properties:
- `analyse`
  - `bool` that determines if audio analysis is conducted on this stream. This can be turned off for certain streams if not needed for effiecency.    
- `amplitude`
  - `float` of the current amplitude 
- `on_new_frame`
  - `callback` called whenever a new audio buffer is passed to the output with the `audio_buffer` as an argument.  
- `channels`
  - `int` set to the `max_output_channels` of the `output_device` 
- `gain`
  - `float` gain of the device 
- `output_device`
  - `int` index of given output device.  A list of ids can be  found by executing `print(sd.query_devices())`.
- `fft_size`
  - `int`  
- `sr`
  - `int` sample rate 
- `buffer_size`
  - `int` 

### Methods:

- `play(self)`
- `pause(self)`
- `resume(self)`
- `stop(self)`

## Class `FilePlayer` (inherits from `AudioDevice`)

### Properties:

- `y`
  - `np.array` of the loaded audio file

### Methods:

No public facing API

## Class `AudioCapture` (inherits from `AudioDevice`)

### Properties:

- `input_buffer`
  - `int` id of audio device being captured
- `ptr`
  - `int` of the current frame being played back (start of buffer) in samples

### Methods:

No public facing API

## Class `MAGNetPlayer` (inherits from `AudioDevice`)

### Properties:

No public facing API

### Methods:

No public facing API

## Class `RAVEPlayer` (inherits from `AudioDevice`)

### Properties:
- `current_latent`
  - Set this to update the z vector 
- `latent_dim`
  - Size of latent space 
- `z_bias`
  - constant bias to add to `current_latent`





