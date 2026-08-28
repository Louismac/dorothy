import numpy as np
import moderngl_window as mglw
import time
import traceback
import sys
import os


class DorothyWindow(mglw.WindowConfig):
    """Internal window configuration for moderngl-window"""
    
    gl_version = (3, 3)
    title = "Dorothy - ModernGL"
    resizable = True
    # cursor = True  # Enable cursor tracking
    samples = 4  # Enable MSAA for smoother lines
    vsync = True
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        from .renderer import DorothyRenderer

        from .Dorothy import Dorothy
        from .Audio import Audio
        # Get the Dorothy instance that's waiting for us
        self.dorothy = Dorothy._pending_instance
        self.start_time_millis = int(round(time.time() * 1000))
        # Setup renderer with the context
        fbo_width, fbo_height = self.wnd.fbo.size  # Use framebuffer size!

        self.dorothy.renderer = DorothyRenderer(
            self.ctx, 
            fbo_width,  # Not window_size[0]
            fbo_height  # Not window_size[1]
        )
 
        self.dorothy.wnd = self.wnd
        self.dorothy._initialized = True
        self.dorothy.music = Audio()
        
        # Set default 2D camera mode
        self.dorothy.renderer.camera.mode = '2d'

        self.dorothy.keys = self.wnd.keys
        self.dorothy.modifiers = self.wnd.modifiers

        self._focused = False

        self.dorothy._ensure_persistent_canvas()
            
        # Now ALL drawing in setup goes to the persistent canvas
        self.dorothy.renderer.begin_layer(self.dorothy._persistent_canvas)
        
        # Call user setup
        if self.dorothy.setup_fn:
            self.dorothy.setup_fn()
        
        self.dorothy.renderer.end_layer()
        self._resize_pending = False
        self._last_resize = (0, 0)




    def _request_focus(self):
        try:
            self.wnd._window.activate()
        except Exception:
            pass
        if sys.platform == 'darwin':
            try:
                import subprocess
                subprocess.Popen([
                    'osascript', '-e',
                    f'tell application "System Events" to set frontmost of first process whose unix id is {os.getpid()} to true'
                ])
            except Exception:
                pass

    def on_render(self, render_time: float, frame_time: float):
        """Called every frame"""
        if not self._focused:
            self._request_focus()
            self._focused = True
        # Create persistent canvas if needed
        try:
            self.dorothy._ensure_persistent_canvas()
            
            # Initialize non-accumulating shader output tracker
            self.dorothy._non_accumulating_shader_output = None
            
            # ALL user drawing goes to the persistent canvas
            self.dorothy.renderer.begin_layer(self.dorothy._persistent_canvas)
            self.dorothy.renderer.transform.reset()
            
            # Call user draw function
            try:
                if self.dorothy.draw_fn:
                    self.dorothy.draw_fn()
            except Exception as e:
                if self.dorothy.frames < 5:
                    print(f"Error in draw(): {e}")
                    if self.dorothy.frames == 0:
                        traceback.print_exc()
            
            self.dorothy.renderer.flush_batch()
            
            # End drawing to persistent canvas
            self.dorothy.renderer.end_layer()
            
            # Clear screen
            self.ctx.screen.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            
            # Display either shader output OR persistent canvas
            if self.dorothy._non_accumulating_shader_output is not None:
                # Non-accumulating shader was used - display its output
                self.dorothy.renderer.draw_layer(self.dorothy._non_accumulating_shader_output)
                
                # Clean up the temporary shader output
                temp_layer = self.dorothy.renderer.layers[self.dorothy._non_accumulating_shader_output]
                temp_layer['fbo'].release()
                temp_layer['texture'].release()
                del self.dorothy.renderer.layers[self.dorothy._non_accumulating_shader_output]
            else:
                # Normal mode - display persistent canvas
                self.dorothy.renderer.draw_layer(self.dorothy._persistent_canvas)
            
            self.end_render()
            
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.dorothy.exit()  

    def end_render(self):

        # Use the actual resized dimensions, not the FBO size
        if self._resize_pending:
            width, height = self._last_resize
            if self.dorothy.renderer:
                print(f"Window resize finished, updating to {width} x {height}")

                # Grab the old canvas's pixels before we tear anything down,
                # so whatever setup()/draw() had already painted survives.
                old_canvas = self.dorothy._persistent_canvas
                old_pixels = None
                if old_canvas is not None:
                    old_pixels = self.dorothy.renderer.get_pixels(
                        layer_id=old_canvas, components=4, flip=True, bgr=False
                    )

                self.dorothy.renderer.width = width
                self.dorothy.renderer.height = height
                self.dorothy.renderer.camera.width = width
                self.dorothy.renderer.camera.height = height
                self.dorothy.renderer.camera.aspect = width / height
                self.ctx.viewport = (0, 0, width, height)

                # Recreate the persistent canvas at the new resolution.
                # The old FBO texture is still the original size and would
                # render incorrectly into the enlarged viewport, so we
                # rebuild it and paste the captured content back in
                # (top-left anchored, unscaled) instead of leaving it blank.
                if old_canvas is not None:
                    self.dorothy.renderer.release_layer(old_canvas)
                    self.dorothy._persistent_canvas = self.dorothy.renderer.get_layer()
                    if old_pixels is not None:
                        prev_camera_mode = self.dorothy.renderer.camera.mode
                        self.dorothy.renderer.camera.mode = '2d'
                        self.dorothy.renderer.begin_layer(self.dorothy._persistent_canvas)
                        self.dorothy.paste(old_pixels, (0, 0), size=(width, height))
                        self.dorothy.renderer.end_layer()
                        self.dorothy.renderer.camera.mode = prev_camera_mode
            self._resize_pending = False
        
        self.dorothy.frames += 1
        self.dorothy.update_lfos()
        
        if self.dorothy.recording:
            canvas_rgb = self.dorothy.renderer.get_pixels()
            self.dorothy.video_recording_buffer.append({
                "frame": canvas_rgb,
                "timestamp": self.dorothy.millis
            })

        if self.dorothy.recording and self.dorothy.end_recording_at < self.dorothy.millis:
            try:
                self.dorothy.stop_record()
            except Exception as e:
                print("error recording video")
                print(e)
                traceback.print_exc()
                self.dorothy.exit() 
            self.dorothy.end_recording_at = np.inf

    def on_mouse_position_event(self, x, y, dx, dy):
        # print(f"mouse pos {x} {y}")
        self.dorothy.mouse_x = int(x)
        self.dorothy.mouse_y = int(y)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.dorothy.mouse_x = int(x)
        self.dorothy.mouse_y = int(y)
        if self.dorothy.on_mouse_drag is not None:
            self.dorothy.on_mouse_drag(x,y,dx,dy)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        if self.dorothy.on_scroll is not None:
            self.dorothy.on_scroll(x_offset,y_offset)
            print("Mouse wheel:", x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        # Update buttons immediately on press
        self.dorothy.mouse_pressed = True
        self.dorothy.mouse_button = button
        if self.dorothy.on_mouse_press is not None:
            self.dorothy.on_mouse_press(x,y,button)
            print("Mouse button {} pressed at {}, {}".format(button, x, y))

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.dorothy.mouse_pressed = False
        if self.dorothy.on_mouse_release is not None:
            self.dorothy.on_mouse_release(x,y,button)
            print("Mouse button {} released at {}, {}".format(button, x, y))

    
    def on_key_event(self, key, action, modifiers):
        print(key, action, modifiers)
        if self.dorothy.on_key_press is not None:
            self.dorothy.on_key_press(key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Q or key == self.wnd.keys.ESCAPE:
                self.wnd.close()
                print("Window closing...")
    
    def on_close(self):
        """Called when window is closing"""
        # Call user cleanup callback
        print("close!!!")
        if self.dorothy.on_close:
            try:
                self.dorothy.on_close()
            except Exception as e:
                print(f"Error in on_close callback: {e}")
        self.dorothy.exit()  
    
    
    def on_resize(self, width: int, height: int):

        # Window activation/focus can trigger a spurious resize event with
        # the same dimensions (e.g. on macOS when we force the window
        # frontmost). Ignore it so we don't needlessly wipe the persistent
        # canvas and lose whatever setup() drew.
        if self.dorothy.renderer is not None and \
           (width, height) == (self.dorothy.renderer.width, self.dorothy.renderer.height):
            return

        # Store the resize but don't process immediately during drag
        print(f"Window was resized to {width} x {height}")
        self._last_resize = (width, height)
        self._resize_pending = True
            
        
