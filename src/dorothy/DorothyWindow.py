import numpy as np
import moderngl_window as mglw
import time
import traceback

class DorothyWindow(mglw.WindowConfig):
    """Internal window configuration for moderngl-window"""
    
    gl_version = (3, 3)
    title = "Dorothy - ModernGL"
    # resizable = True
    cursor = True  # Enable cursor tracking
    samples = 4  # Enable MSAA for smoother lines
    # vsync = True
    
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

    
        self.dorothy._ensure_persistent_canvas()
            
        # Now ALL drawing in setup goes to the persistent canvas
        self.dorothy.renderer.begin_layer(self.dorothy._persistent_canvas)
        
        # Call user setup
        if self.dorothy.setup_fn:
            self.dorothy.setup_fn()
        
        self.dorothy.renderer.end_layer()
       


    def on_render(self, render_time: float, frame_time: float):
        """Called every frame"""
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
        if self.dorothy.on_mouse_drag is not None:
            self.dorothy.on_mouse_drag(x,y,dx,dy)
            print("Mouse drag:", x, y, dx, dy)

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

    
    def resize(self, width: int, height: int):
        print(f"\n=== RESIZE EVENT ===")
        print(f"Window size: {width}x{height}")
        print(f"window_size attr: {self.window_size}")
        print(f"FBO size: {self.wnd.fbo.size}")
        print(f"Renderer size: {self.dorothy.renderer.width}x{self.dorothy.renderer.height}")
        print(f"Camera size: {self.dorothy.renderer.camera.width}x{self.dorothy.renderer.camera.height}")
        print(f"Viewport: {self.ctx.viewport}")
        fbo_width, fbo_height = self.wnd.fbo.size
        if self.dorothy.renderer:
            self.dorothy.renderer.width = fbo_width
            self.dorothy.renderer.height = fbo_height
            self.dorothy.renderer.camera.width = fbo_width
            self.dorothy.renderer.camera.height = fbo_height
            self.dorothy.renderer.camera.aspect = fbo_width / fbo_height
