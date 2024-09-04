
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

- ## Class `Audio`

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
-   

### Methods:
- `on_analysis_complete(self, fft_vals, amplitude)`
- `start_magnet_stream(self, model_path, dataset_path, buffer_size=2048, sr=44100, output_device=None)`

## Class `AudioDevice`

### Properties:
- `analyse`
- `amplitude`
- `on_analysis_complete`
- `on_new_frame`
- `channels`
- `play_thread`
- `gain`
- `running`
- `output_device`
- `fft_size`
- `sr`
- `buffer_size`
- `pause_event`
- `audio_callback`
- `internal_callback`
- `fft_vals`

### Methods:
- `__init__(self, on_analysis_complete=<expression>, on_new_frame=<expression>, analyse=True, fft_size=1024, buffer_size=2048, sr=44100, output_device=None)`
- `do_analysis(self, audio_buffer)`
- `audio_callback(self)`
- `capture_audio(self)`
- `play(self)`
- `pause(self)`
- `resume(self)`
- `stop(self)`

## Class `MAGNetPlayer` (inherits from `AudioDevice`)

### Properties:
- `running`
- `model`
- `sequence_length`
- `x_frames`
- `current_buffer`

### Methods:
- `__init__(self, on_analysis_complete=<expression>, on_new_frame=<expression>, analyse=True, fft_size=1024, buffer_size=2048, sr=44100, output_device=None)`
- `load_model(self, path)`
- `load_dataset(self, path)`
- `skip(self, index=0)`
- `fill_next_buffer(self)`
- `get_frame(self)`
- `audio_callback(self)`

## Class `RAVEPlayer` (inherits from `AudioDevice`)

### Properties:
- `current_latent`
- `latent_dim`
- `z_bias`





