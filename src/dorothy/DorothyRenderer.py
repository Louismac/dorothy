from .Audio import *
from .DorothyWindow import *
from .DorothyRenderer import *
import numpy as np
import moderngl
from moderngl_window import geometry
import glm
from typing import Tuple, Optional
import cv2
from .DorothyShaders import DOTSHADERS


class Transform:
    """Manages transformation matrices""" 
    def __init__(self):
        self.matrix = glm.mat4(1.0)  # Identity matrix
        self.stack = []
        
    
    def push(self):
        self.stack.append(glm.mat4(self.matrix))
    
    def pop(self):
        if self.stack:
            self.matrix = self.stack.pop()
    
    def reset(self):
        self.matrix = glm.mat4(1.0)
    
    def translate(self, x: float, y: float, z: float = 0):
        """Translate"""
        self.matrix = glm.translate(self.matrix, glm.vec3(x, y, z))
    
    def rotate(self, angle: float, x: float = 0, y: float = 0, z: float = 1):
        """Rotate around an axis
        
        Args:
            angle: Rotation angle in radians
            x, y, z: Rotation axis (default: z-axis for 2D rotation)
        """
        self.matrix = glm.rotate(self.matrix, angle, glm.vec3(x, y, z))
    
    def scale(self, x: float, y: float = None, z: float = None):
        """Scale
        
        Args:
            x: Scale factor for x-axis (if y and z are None, scales uniformly)
            y: Scale factor for y-axis (default: same as x)
            z: Scale factor for z-axis (default: same as x)
        """
        y = y if y is not None else x
        z = z if z is not None else x
        self.matrix = glm.scale(self.matrix, glm.vec3(x, y, z))


class Camera:
    """3D Camera with perspective and orthographic modes"""
    def __init__(self, width: int, height: int):
        self.position = glm.vec3(0, 0, 5)
        self.target = glm.vec3(0, 0, 0)
        self.up = glm.vec3(0, 1, 0)
        self.fov = 60
        self.aspect = width / height
        self.near = 0.1
        self.far = 100.0
        self.mode = '3d'  # '3d' or '2d'
        self.width = width
        self.height = height
    
    def get_view_matrix(self):
        return glm.lookAt(self.position, self.target, self.up)
    
    def get_projection_matrix(self):
        if self.mode == '3d':
            return glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
        else:
            # Orthographic projection for 2D mode
            return glm.ortho(0, self.width, self.height, 0, -1, 1)


class DorothyRenderer:
    """Core rendering engine using ModernGL"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # State
        self.fill_color = (255, 255, 255, 255)
        self.stroke_color = (0, 0, 0, 255)
        self._stroke_weight = 1
        self.use_fill = True
        self.use_stroke = False
        
        # Transform and camera
        self.transform = Transform()
        self.camera = Camera(width, height)
        
        # Setup shaders
        self._setup_shaders()
        
        # Geometry cache
        self._setup_geometry()
        
        # Background
        self.background_color = (0, 0, 0, 1)
        
        # Layer system
        self.layers = {}  # Dictionary of layer_id -> framebuffer
        self.layer_stack = []
        self.layer_counter = 0
        self.active_layer = None  # Currently rendering to a layer
        self.effect_shaders = {}
        self.effect_vaos = {}
        
    def _setup_shaders(self):
        """Initialize shader programs"""
        
        # Basic 3D shader with lighting
        self.shader_3d = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D,
            fragment_shader=DOTSHADERS.FRAG_3D
        )
        
        # Simple 2D shader
        self.shader_2d = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D,
            fragment_shader=DOTSHADERS.FRAG_2D
        )
        
# Texture shader for rendering layers
        self.shader_texture = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_TEXTURE,
            fragment_shader=DOTSHADERS.FRAG_TEXTURE
        )
        
        # Texture shader with transforms for layer positioning
        self.shader_texture_transform = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_TEXTURE_TRANSFORM,
            fragment_shader=DOTSHADERS.FRAG_TEXTURE_TRANSFORM
        )
        

        self.shader_texture_2d = self.ctx.program(
                vertex_shader=DOTSHADERS.VERT_TEXTURE_2D,
                fragment_shader=DOTSHADERS.FRAG_TEXTURE_2D
            )
        
        # Create fullscreen quad for texture rendering
        vertices = np.array([
            # Position   # TexCoord
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0, -1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
        ], dtype='f4')
        
        self.quad_vbo = self.ctx.buffer(vertices)
        self.quad_vao = self.ctx.simple_vertex_array(
            self.shader_texture, 
            self.quad_vbo, 
            'in_position', 'in_texcoord'
        )
        
    def _setup_geometry(self):
        """Setup reusable geometry"""
        # 3D primitives will be created on-demand using moderngl_window.geometry
        self.sphere_geometry = None
        self.box_geometry = None
        
    def clear(self, color: Optional[Tuple] = None):
        """Clear the screen"""
        if color:
            self.background_color = self._normalize_color(color)
        self.ctx.clear(*self.background_color)
    
    
    
    # ====== Layer Management ======
    
    def get_layer(self) -> int:
        """Create a new layer (framebuffer) and return its ID
        
        Returns:
            layer_id: Unique identifier for this layer
        """
        layer_id = self.layer_counter
        self.layer_counter += 1
        
        
        # Create texture for this layer with alpha channel
        texture = self.ctx.texture((self.width, self.height), 4)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        
        # Create depth buffer for the framebuffer
        depth = self.ctx.depth_renderbuffer((self.width, self.height))
        
        # Create framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=[texture],
            depth_attachment=depth
        )
        
        # Store layer
        self.layers[layer_id] = {
            'fbo': fbo,
            'texture': texture,
            'depth': depth
        }
        
        # Clear the layer initially
        # fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        # self.ctx.screen.use()
        
        print(f"Created layer {layer_id}: {self.width}x{self.height}")
        
        return layer_id
    
    def begin_layer(self, layer_id: int):
        """Start rendering to a specific layer"""
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        # Push current layer onto stack
        self.layer_stack.append(self.active_layer)
        
        self.active_layer = layer_id
        fbo = self.layers[layer_id]['fbo']
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)


    def end_layer(self):
        """Stop rendering to layer, return to previous layer"""
        if self.active_layer is None:
            return
        
        # Pop previous layer from stack
        if self.layer_stack:
            prev_layer = self.layer_stack.pop()
            
            if prev_layer is not None:
                # Return to previous layer
                self.active_layer = prev_layer
                self.layers[prev_layer]['fbo'].use()
            else:
                # Return to screen
                self.active_layer = None
                self.ctx.screen.use()
        else:
            # No stack, default to screen
            self.active_layer = None
            self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

    def draw_layer(self, layer_id: int, alpha: float = 1.0, x: int = 0, y: int = 0):
        """Draw a layer to the current render target with optional transparency
        
        Args:
            layer_id: The layer to draw
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)
            x, y: Position offset (TODO: implement positioning)
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        # Don't change render target - draw to whatever is currently active
        # (This will be the persistent canvas during user's draw loop)
        self.ctx.disable(moderngl.DEPTH_TEST)
        # Enable proper alpha blending for layer compositing
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA
        )
        self.ctx.depth_func = '<'
        
        # Bind the layer's texture
        texture = self.layers[layer_id]['texture']
        texture.use(0)
        
        # Check if we have active transforms
        has_transform = not np.allclose(self.transform.matrix, glm.mat4(1.0))
        
        if has_transform:
            # Use transform-aware shader with screen-space coordinates
            shader = self.shader_texture_transform
            
            vertices = np.array([
                # Position (screen coords)  # TexCoord (V flipped!)
                0, 0,                        0.0, 1.0,  # Top-left
                self.width, 0,               1.0, 1.0,  # Top-right
                self.width, self.height,     1.0, 0.0,  # Bottom-right
                0, 0,                        0.0, 1.0,  # Top-left
                self.width, self.height,     1.0, 0.0,  # Bottom-right
                0, self.height,              0.0, 0.0,  # Bottom-left
            ], dtype='f4')
                        
            vbo = self.ctx.buffer(vertices)
            vao = self.ctx.simple_vertex_array(shader, vbo, 'in_position', 'in_texcoord')
            
            # Set uniforms
            shader['projection'].write(self.camera.get_projection_matrix())
            shader['model'].write(self.transform.matrix)
            shader['texture0'] = 0
            shader['alpha'] = alpha
            
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            # Use simple fullscreen shader (NDC coordinates)
            shader = self.shader_texture
            
            # Fullscreen quad in NDC
            shader['texture0'] = 0
            shader['alpha'] = alpha
            self.quad_vao.render(moderngl.TRIANGLES)
                
    
    def release_layer(self, layer_id: int):
        """Free a layer's resources
        
        Args:
            layer_id: The layer to release
        """
        if layer_id in self.layers:
            self.layers[layer_id]['fbo'].release()
            self.layers[layer_id]['texture'].release()
            del self.layers[layer_id]
    
    def clear_layer(self, layer_id: int, color: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        """Clear a layer with a specific color
        
        Args:
            layer_id: The layer to clear
            color: RGBA color (0.0-1.0 range)
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        fbo = self.layers[layer_id]['fbo']
        fbo.use()
        fbo.clear(*color)
    
    # ====== Image/Texture Pasting ======

    def get_effect_shader(self, shader_code):
        shader_hash = hash(shader_code)
        """Get or create a cached shader program"""
        if shader_hash not in self.effect_shaders.keys():
            print(f"Compiling shader...")
            print(f"Shader code:\n{shader_code}")
            
            vertex_shader = '''
                #version 330
                
                in vec2 in_position;
                in vec2 in_texcoord;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            '''
            
            try:
                program = self.ctx.program(
                    vertex_shader=vertex_shader,
                    fragment_shader=shader_code
                )
                
                print(f"Compilation result type: {type(program)}")
                self.effect_shaders[shader_hash] = program
                if not hasattr(program, '__getitem__'):
                    print("Program doesn't have required methods")
                    return None
                    
            except Exception as e:
                print(f"Shader compilation exception: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return self.effect_shaders[shader_hash]

    def apply_shader(self, fragment_shader_code: str, uniforms: dict = None, accumulate: bool = True):
        """Apply a custom fragment shader to the current canvas
        
        Args:
            fragment_shader_code: GLSL fragment shader code
            uniforms: Optional dict of additional uniforms to set
            accumulate: If True, shader modifies persistent canvas (feedback effects)
                    If False, shader output is shown but not fed back (post-processing)
        """
        # Get cached shader (or compile if first time)
        custom_shader = self.get_effect_shader(fragment_shader_code)
        if custom_shader is None:
            print("Shader is None, skipping effect")
            return None
        # Verify it's a real shader program
        if not hasattr(custom_shader, 'get'):
            print(f"Invalid shader object: {type(custom_shader)}")
            return None
        # Get current layer
        layer = self.layers[self.active_layer]
        old_texture = layer['texture']
        old_fbo = layer['fbo']
        # Create NEW texture and FBO for the shader output
        new_texture = self.ctx.texture((self.width, self.height), 4)
        new_fbo = self.ctx.framebuffer(color_attachments=[new_texture])
        # Get or create cached VAO for this shader
        shader_hash = hash(fragment_shader_code)
        if shader_hash not in self.effect_vaos:
            print("making vao (caching)")
            self.effect_vaos[shader_hash] = self.ctx.simple_vertex_array(
                custom_shader,
                self.quad_vbo,
                'in_position',
                'in_texcoord'
            )
        
        custom_vao = self.effect_vaos[shader_hash]
        # Render to new FBO with custom shader
        new_fbo.use()
        old_texture.use(0)
        
        # Set uniforms
        try:
            custom_shader['texture0'] = 0
        except KeyError:
            pass
        
        try:
            custom_shader['resolution'] = (float(self.width), float(self.height))
        except KeyError:
            pass
        
        if uniforms:
            for name, value in uniforms.items():
                try:
                    if isinstance(value, (int, float)):
                        custom_shader[name] = float(value)
                    elif isinstance(value, (tuple, list)):
                        custom_shader[name] = tuple(float(v) for v in value)
                    else:
                        custom_shader[name] = value
                except KeyError:
                    pass
        self.ctx.disable(moderngl.BLEND)
        custom_vao.render(moderngl.TRIANGLES)
        
        if accumulate:
            # ACCUMULATING MODE: Replace persistent canvas with shader output
            # Shader effects build up over frames
            layer['texture'] = new_texture
            layer['fbo'] = new_fbo
            
            # Clean up old resources
            old_fbo.release()
            old_texture.release()
            
            # Set new FBO as active
            new_fbo.use()
            
            return None
        else:
            # NON-ACCUMULATING MODE: Keep persistent canvas unchanged
            # Return the shader output to be displayed instead
            
            # Restore the original FBO as active (keep persistent canvas intact)
            old_fbo.use()
            
            # Return a temporary layer ID that on_render can display
            # Store it temporarily
            temp_layer_id = -1  # Special ID for non-accumulating shader output
            self.layers[temp_layer_id] = {
                'fbo': new_fbo,
                'texture': new_texture,
                'depth': None
            }
            
            # Re-enable blending
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            
            return temp_layer_id

    def get_pixels(self) -> np.ndarray:
        """Get current screen pixels as numpy array
        
        Returns:
            np.ndarray: RGB image array (height, width, 3) in uint8 format
        """
        # Read pixels from the screen framebuffer
        pixels = self.ctx.screen.read(components=3)
        
        # Convert to numpy array
        img = np.frombuffer(pixels, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 3))
        
        # Flip vertically (OpenGL origin is bottom-left)
        img = np.flipud(img)
        
        # Convert RGB to BGR for OpenCV compatibility
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def paste(self, image: np.ndarray, position: Tuple[int, int], 
              size: Optional[Tuple[int, int]] = None, alpha: float = 1.0):
        """Paste a numpy array (image) onto the canvas
        
        Args:
            image: NumPy array of pixels. Can be:
                   - (H, W, 3) for RGB
                   - (H, W, 4) for RGBA
                   - (H, W) for grayscale
            position: (x, y) position to paste (top-left corner)
            size: Optional (width, height) to resize image. If None, uses original size
            alpha: Overall transparency (0.0 = invisible, 1.0 = opaque)
        """
        # Normalize image array
        img = self._prepare_image_array(image)
        h, w = img.shape[:2]
        
        # Determine target size
        if size is None:
            target_w, target_h = w, h
        else:
            target_w, target_h = size
                
        # Create texture
        texture = self.ctx.texture((w, h), 4, img.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Use the 2D shader with projection and transform
        x, y = position
        
        # Create vertices in pixel coordinates (6 vertices for 2 triangles)
        # Format: x, y, u, v for each vertex
        # Standard texture coords: (0,0) = bottom-left, (1,1) = top-right in OpenGL
        # But images are top-left origin, so we flip V: top=0, bottom=1
        vertices = np.array([
            # Triangle 1
            x, y, 0.0, 0.0,                           # Top-left
            x + target_w, y, 1.0, 0.0,                # Top-right
            x + target_w, y + target_h, 1.0, 1.0,     # Bottom-right
            # Triangle 2
            x, y, 0.0, 0.0,                           # Top-left
            x + target_w, y + target_h, 1.0, 1.0,     # Bottom-right
            x, y + target_h, 0.0, 1.0                 # Bottom-left
        ], dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
            
        vao = self.ctx.simple_vertex_array(
            self.shader_texture_2d,
            vbo,
            'in_position', 'in_texcoord'
        )
        
        # Set uniforms - use current transform and camera
        self.shader_texture_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_texture_2d['model'].write(self.transform.matrix)
        self.shader_texture_2d['texture0'] = 0
        self.shader_texture_2d['alpha'] = alpha
        
        
        # Bind texture and render
        texture.use(0)
        vao.render(moderngl.TRIANGLES)
        
        
        # Cleanup
        vao.release()
        vbo.release()
        texture.release()
    
    def _prepare_image_array(self, image: np.ndarray) -> np.ndarray:
        """Prepare image array for OpenGL texture
        
        Converts various image formats to RGBA uint8
        """
        img = np.asarray(image)
        
        # Handle different input formats
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Assume 0-1 range, convert to 0-255
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Handle different channel counts
        if len(img.shape) == 2:
            # Grayscale -> RGBA
            h, w = img.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = img
            rgba[:, :, 1] = img
            rgba[:, :, 2] = img
            rgba[:, :, 3] = 255
            img = rgba
        elif img.shape[2] == 3:
            # RGB -> RGBA
            h, w = img.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = 255
            img = rgba
        elif img.shape[2] == 4:
            # Already RGBA
            pass
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        
        return img
        
    def _normalize_color(self, color: Tuple) -> Tuple[float, float, float, float]:
        """Convert color from 0-255 to 0-1 range"""
        if len(color) == 3:
            return (color[0]/255, color[1]/255, color[2]/255, 1.0)
        elif len(color) == 4:
            return (color[0]/255, color[1]/255, color[2]/255, color[3]/255)
        else:
            raise ValueError(f"Color must be RGB or RGBA tuple, got: {color}")
    
    def _draw_annotation(self, position: Tuple[float, float], text: str):
        """Draw annotation text near a shape (simplified version)
        
        Note: Full text rendering requires a font atlas. This is a placeholder
        that draws a small indicator. For production, integrate a text rendering
        library like moderngl-text or PIL-based texture text.
        """
        # Draw a small cross at the position to indicate annotation point
        x, y = position
        offset = 3
        
        # Save current stroke settings
        old_stroke = self.use_stroke
        old_color = self.stroke_color
        old_weight = self._stroke_weight
        
        # Draw cross
        self.use_stroke = True
        self.stroke_color = (255, 255, 0, 255)  # Yellow
        self._stroke_weight = 1
        
        # Vertical line
        vertices = np.array([x, y - offset, x, y + offset], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
        vao.render(moderngl.LINES)
        vao.release()
        vbo.release()
        
        # Horizontal line
        vertices = np.array([x - offset, y, x + offset, y], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
        vao.render(moderngl.LINES)
        vao.release()
        vbo.release()
        
        # Restore stroke settings
        self.use_stroke = old_stroke
        self.stroke_color = old_color
        self._stroke_weight = old_weight
        
        # TODO: Actual text rendering would go here
        # For now, just the cross indicator shows where the annotation would be
    
    def _create_circle_vertices(self, center: Tuple[float, float], radius: float, segments: int = 32):
        """Generate circle vertices"""
        vertices = [center[0], center[1]]  # Center point first for TRIANGLE_FAN
        for i in range(segments + 1):  # +1 to close the circle
            angle = 2 * np.pi * i / segments
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            vertices.extend([x, y])
        return np.array(vertices, dtype='f4')
    
    # ====== 2D Drawing Methods (Processing-like API) ======
    
    def circle(self, center: Tuple[float, float], radius: float, annotate: bool = False):
        """Draw a circle in 2D mode"""
        vertices = self._create_circle_vertices(center, radius)
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Set uniforms
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        
        # Draw fill
        if self.use_fill:
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
            vao.render(moderngl.TRIANGLE_FAN)
        
        # Draw stroke (use LINE_STRIP for circle outline)
        if self.use_stroke:
            # Create separate vertices for stroke (without center point)
            stroke_vertices = []
            segments = 32
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                stroke_vertices.extend([x, y])
            
            stroke_vbo = self.ctx.buffer(np.array(stroke_vertices, dtype='f4'))
            stroke_vao = self.ctx.simple_vertex_array(self.shader_2d, stroke_vbo, 'in_position')
            
            self.ctx.line_width = self._stroke_weight
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
            stroke_vao.render(moderngl.LINE_STRIP)
            
            stroke_vao.release()
            stroke_vbo.release()
        
        vao.release()
        vbo.release()
        
        # Draw annotation if requested
        if annotate:
            self._draw_annotation(center, f"({int(center[0])}, {int(center[1])})\nr={int(radius)}")
    
    def rectangle(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a rectangle"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        vertices = np.array([
            x1, y1,
            x2, y1,
            x2, y2,
            x1, y2
        ], dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        
        if self.use_fill:
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
            vao.render(moderngl.TRIANGLE_FAN)
        
        if self.use_stroke:
            self.ctx.line_width = self._stroke_weight
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
            vao.render(moderngl.LINE_LOOP)
        
        vao.release()
        vbo.release()
        
        # Draw annotation if requested
        if annotate:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self._draw_annotation((center_x, center_y), 
                                f"({int(x1)}, {int(y1)})\n({int(x2)}, {int(y2)})")
    
    def line(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a line"""
        vertices = np.array([
            pos1[0], pos1[1],
            pos2[0], pos2[1]
        ], dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')

        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        self.ctx.line_width = self._stroke_weight
        self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
        
        vao.render(moderngl.LINES)
        
        vao.release()
        vbo.release()
        
        # Draw annotation if requested
        if annotate:
            center_x = (pos1[0] + pos2[0]) / 2
            center_y = (pos1[1] + pos2[1]) / 2
            self._draw_annotation((center_x, center_y), 
                                f"({int(pos1[0])}, {int(pos1[1])})\n({int(pos2[0])}, {int(pos2[1])})")
            
    def polyline(self, points, closed: bool = False):
        """Draw a polyline (connected line segments)
        
        Args:
            points: List of (x, y) coordinates
            closed: If True, connect last point back to first point
        """
        if len(points) < 2:
            return  # Need at least 2 points for a line
        
        # Create vertices for the line strip
        vertices = []
        for x, y in points:
            vertices.extend([x, y])
        
        # If closed, add the first point again at the end
        if closed:
            vertices.extend([points[0][0], points[0][1]])
        
        vertices = np.array(vertices, dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Set uniforms
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        
        # Draw stroke
        if self.use_stroke:
            self.ctx.line_width = self._stroke_weight
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
            vao.render(moderngl.LINE_STRIP)
        
        vao.release()
        vbo.release()

    def polygon(self, points):
        """Draw a filled polygon with proper triangulation
        
        Args:
            points: List of (x, y) coordinates defining the polygon vertices
        """
        if len(points) < 3:
            return  # Need at least 3 points for a polygon
        
        # Triangulate the polygon using ear clipping algorithm
        def triangulate(vertices):
            """Simple ear clipping triangulation for polygons"""
            if len(vertices) < 3:
                return []
            
            # Make a copy to work with
            verts = list(vertices)
            triangles = []
            
            while len(verts) > 3:
                # Find an ear (a triangle that doesn't contain other vertices)
                ear_found = False
                for i in range(len(verts)):
                    prev = verts[i - 1]
                    curr = verts[i]
                    next_v = verts[(i + 1) % len(verts)]
                    
                    # Check if this is a valid ear
                    if is_ear(prev, curr, next_v, verts):
                        triangles.extend([prev, curr, next_v])
                        verts.pop(i)
                        ear_found = True
                        break
                
                if not ear_found:
                    # Fallback: just use remaining vertices as one triangle
                    if len(verts) >= 3:
                        triangles.extend(verts[:3])
                    break
            
            # Add the last triangle
            if len(verts) == 3:
                triangles.extend(verts)
            
            return triangles
        
        def is_ear(prev, curr, next_v, all_verts):
            """Check if the triangle (prev, curr, next) is an ear"""
            # Check if angle at curr is convex
            cross = (curr[0] - prev[0]) * (next_v[1] - prev[1]) - (curr[1] - prev[1]) * (next_v[0] - prev[0])
            if cross <= 0:  # Reflex angle
                return False
            
            # Check if any other vertex is inside this triangle
            for v in all_verts:
                if v == prev or v == curr or v == next_v:
                    continue
                if point_in_triangle(v, prev, curr, next_v):
                    return False
            
            return True
        
        def point_in_triangle(p, a, b, c):
            """Check if point p is inside triangle abc"""
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
            d1 = sign(p, a, b)
            d2 = sign(p, b, c)
            d3 = sign(p, c, a)
            
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            
            return not (has_neg and has_pos)
        
        # Triangulate the polygon
        triangulated = triangulate(points)
        
        if not triangulated:
            return
        
        # Create vertices array from triangulated points
        vertices = []
        for x, y in triangulated:
            vertices.extend([x, y])
        
        vertices = np.array(vertices, dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Set uniforms
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        
        # Draw fill
        if self.use_fill:
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
            vao.render(moderngl.TRIANGLES)
        
        # Draw stroke
        if self.use_stroke:
            # Create separate vertices for stroke outline
            stroke_vertices = []
            for x, y in points:
                stroke_vertices.extend([x, y])
            # Close the loop
            stroke_vertices.extend([points[0][0], points[0][1]])
            
            stroke_vbo = self.ctx.buffer(np.array(stroke_vertices, dtype='f4'))
            stroke_vao = self.ctx.simple_vertex_array(self.shader_2d, stroke_vbo, 'in_position')
            
            self.ctx.line_width = self._stroke_weight
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
            stroke_vao.render(moderngl.LINE_STRIP)
            
            stroke_vao.release()
            stroke_vbo.release()
        
        vao.release()
        vbo.release()
    # ====== 3D Drawing Methods ======
    
    def sphere(self, radius: float = 1.0, position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D sphere
        
        Args:
            radius: Sphere radius
            position: (x, y, z) center position
        """
        if not self.sphere_geometry:
            self.sphere_geometry = geometry.sphere(radius=1.0, sectors=32, rings=32)
        
        # Apply position and scale
        self.transform.push()
        self.transform.translate(position[0], position[1], position[2])
        self.transform.scale(radius)
        
        self._draw_3d_geometry(self.sphere_geometry)
        
        self.transform.pop()
    
    def box(self, size: Tuple[float, float, float] = (1.0, 1.0, 1.0), 
            position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D box
        
        Args:
            size: (width, height, depth) tuple
            position: (x, y, z) center position
        """
        width, height, depth = size
        
        if not self.box_geometry:
            self.box_geometry = geometry.cube(size=(1.0, 1.0, 1.0))
        
        # Apply position and scale
        self.transform.push()
        self.transform.translate(position[0], position[1], position[2])
        self.transform.scale(width, height, depth)
        
        self._draw_3d_geometry(self.box_geometry)
        
        self.transform.pop()
    
    def _draw_3d_geometry(self, geom):
        """Internal method to render 3D geometry"""
        self.shader_3d['model'].write(self.transform.matrix)
        self.shader_3d['view'].write(self.camera.get_view_matrix())
        self.shader_3d['projection'].write(self.camera.get_projection_matrix())
        self.shader_3d['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
        self.shader_3d['light_pos'].write(glm.vec3(5, 5, 5))
        self.shader_3d['camera_pos'].write(self.camera.position)
        self.shader_3d['use_lighting'] = True
        
        geom.render(self.shader_3d)
    
    # ====== State Methods ======
    
    def fill(self, color: Tuple):
        """Set fill color"""
        # Handle both RGB and RGBA
        if len(color) == 3:
            self.fill_color = (*color, 255)
        else:
            self.fill_color = color
        self.use_fill = True
    
    def no_fill(self):
        """Disable fill"""
        self.use_fill = False
    
    def stroke(self, color: Tuple):
        """Set stroke color"""
        # Handle both RGB and RGBA
        if len(color) == 3:
            self.stroke_color = (*color, 255)
        else:
            self.stroke_color = color
        self.use_stroke = True
    
    def no_stroke(self):
        """Disable stroke"""
        self.use_stroke = False
    
    def set_stroke_weight(self, weight: float):
        """Set stroke weight"""
        self._stroke_weight = weight
    
    # ====== Transform Methods ======
    
    def push_matrix(self):
        """Save current transformation"""
        self.transform.push()
    
    def pop_matrix(self):
        """Restore previous transformation"""
        self.transform.pop()
    
    def translate(self, x: float, y: float, z: float = 0):
        """Translate"""
        self.transform.translate(x, y, z)
    
    def rotate(self, angle: float, x: float = 0, y: float = 0, z: float = 1):
        """Rotate (angle in radians)"""
        self.transform.rotate(angle, x, y, z)
    
    def scale(self, x: float, y: float = None, z: float = None):
        """Scale"""
        self.transform.scale(x, y, z)
    
    def reset_transforms(self):
        """Reset all transformations"""
        self.transform.reset()