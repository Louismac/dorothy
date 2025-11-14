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
from dataclasses import dataclass
from typing import List, Tuple, Literal
from enum import Enum
from itertools import groupby

class DrawCommandType(Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    LINE = "line"
    TRIANGLE = "triangle"
    LINE_3D = "3dline"
    POLYLINE_3D = "3dpolyline"
    SPHERE = "sphere"
    THICK_LINE_3D = "3dthickline"
    BOX = "box"

@dataclass
class DrawCommand:
    """Represents a single draw command"""
    type: DrawCommandType
    fill_vertices: Optional[np.ndarray] = None  
    stroke_vertices: Optional[np.ndarray] = None 
    color: Optional[Tuple[float, float, float, float]] = None
    use_fill: bool = True
    use_stroke: bool = False
    stroke_weight: float = 1.0
    stroke_color: Optional[Tuple[float, float, float, float]] = None
    transform: np.ndarray = None
    layer_id: Optional[int] = None
    draw_order: int = 0 
    stroke_as_geometry: bool = False
    texture_layer:dict = None
    is_3d: bool = False 
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None

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
        # Batching system
        self.enable_batching = True  # Can disable for debugging
        self.draw_queue = []  # Queue of DrawCommand objects
        
        # Batched buffers (reused across frames)
        self.batch_vbo = None
        self.batch_vao = None
        self.max_batch_vertices = 100000  # Adjust based on needs
        self.draw_order_counter = 0
        self.last_fbo = None  # Track last bound FBO

        # Transform and camera
        self.transform = Transform()
        self.camera = Camera(width, height)
        self.light_pos = (5,5,5)
        self.use_lighting = True
        self.ambient_light = 0.3
        
        # Setup shaders
        self._setup_shaders()
        
        # Geometry cache
        self._setup_geometry()
        self._circle_geometry_cache = {}
        self._unit_circle_vbo_cache = {}
        self._circle_stroke_cache = {}
        self._unit_circle_stroke_vbo_cache = {} 

        
        # Background
        self.background_color = (0, 0, 0, 1)
        
        # Layer system
        self.layers = {}  # Dictionary of layer_id -> framebuffer
        self.layer_stack = []
        self.layer_counter = 0
        self.active_layer = None  # Currently rendering to a layer
        self.effect_shaders = {}
        self.effect_vaos = {}

    def _ensure_fbo(self, target_fbo):
        """Ensure correct FBO is bound, flushing if needed"""
        if target_fbo != self.last_fbo:
            # FBO changed - flush pending draws first
            if self.enable_batching and len(self.draw_queue) > 0:
                self._flush_for_fbo_change()
            
            # Bind new FBO
            if target_fbo is None:
                self.ctx.screen.use()
            else:
                target_fbo.use()
            
            self.last_fbo = target_fbo
    
    def _flush_for_fbo_change(self):
        """Flush batch because FBO is changing"""
        # Render all queued commands to current FBO
        batches = self._group_commands_into_batches()
        for batch in batches:
            self._render_batch(batch)
        
        self.draw_queue.clear()
        self.draw_order_counter = 0
        
    def _setup_shaders(self):
        """Initialize shader programs"""
        
        self.shader_3d_instanced = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D_INSTANCED,
            fragment_shader=DOTSHADERS.FRAG_3D_INSTANCED
        )

        # Basic 3D shader with lighting
        self.shader_3d = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D_TEXTURED,
            fragment_shader=DOTSHADERS.FRAG_3D_TEXTURED
        )

        self.shader_2d_instanced = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D_INSTANCED,
            fragment_shader=DOTSHADERS.FRAG_2D_INSTANCED
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
            'in_position', 'in_texcoord_0'
        )
        
    def _setup_geometry(self):
        """Setup reusable geometry"""
        # 3D primitives will be created on-demand using moderngl_window.geometry
        verts, normals, texcoords = self._generate_sphere_vertices(radius=1.0, sectors=32, rings=32)

# Interleave the data: position, normal, texcoord for each vertex
        vertex_data = []
        num_verts = len(verts) // 3
        for i in range(num_verts):
            vertex_data.extend(verts[i*3:i*3+3])      # position (3 floats)
            vertex_data.extend(normals[i*3:i*3+3])    # normal (3 floats)
            vertex_data.extend(texcoords[i*2:i*2+2])  # texcoord (2 floats)

        vertex_array = np.array(vertex_data, dtype='f4')
        self.sphere_vbo = self.ctx.buffer(vertex_array)
        self.sphere_ibo = None 
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
        self._ensure_fbo(fbo) 
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)


    def end_layer(self):
        """Stop rendering to layer, return to previous layer"""
        if self.active_layer is None:
            return
        
        if self.enable_batching and len(self.draw_queue) > 0:
            self._flush_for_fbo_change()
        
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
                self.last_fbo = None
                self.ctx.screen.use()
        else:
            # No stack, default to screen
            self.active_layer = None
            self.last_fbo = None
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
            vao = self.ctx.simple_vertex_array(shader, vbo, 'in_position', 'in_texcoord_0')
            
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
        if self.enable_batching and len(self.draw_queue) > 0:
            self._flush_for_fbo_change()
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
                in vec2 in_texcoord_0;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord_0;
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
        if self.enable_batching and len(self.draw_queue) > 0:
            self._flush_for_fbo_change()
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
                'in_texcoord_0'
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

    def get_pixels(self, layer_id=None, components=3, flip=True, bgr=True) -> np.ndarray:
        """Get pixels from a framebuffer as numpy array
        
        Args:
            layer_id: Layer ID to read from, None for active layer/screen
            components: 3 for RGB, 4 for RGBA
            flip: If True, flip vertically (OpenGL coords to screen coords)
            bgr: If True, convert RGB to BGR for OpenCV
        
        Returns:
            np.ndarray: Image array (height, width, components) in uint8 format
        """
        # Determine which framebuffer to read from
        if layer_id is not None:
            # Read from specific layer
            if layer_id not in self.layers:
                raise ValueError(f"Layer {layer_id} does not exist")
            fbo = self.layers[layer_id]['fbo']
            pixels = fbo.read(components=components)
            # Get actual FBO dimensions
            w, h = fbo.size
        # elif self.active_layer is not None:
        #     # Read from currently active layer
        #     fbo = self.layers[self.active_layer]['fbo']
        #     pixels = fbo.read(components=components)
        else:
            # Read from screen
            # print("Reading screen")
            pixels = self.ctx.screen.read(components=components)
            w, h = self.ctx.screen.size
        # Convert to numpy array
        img = np.frombuffer(pixels, dtype=np.uint8)
        
        
        # Convert to numpy array
        img = np.frombuffer(pixels, dtype=np.uint8)
        
        # Reshape using FBO dimensions (not self.width/height which might be swapped)
        img = img.reshape((h, w, components))
        # Flip vertically if requested
        if flip:
            img = np.flipud(img)
        
        # Convert to BGR if requested (for OpenCV)
        if bgr and components == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif bgr and components == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        
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
            'in_position', 'in_texcoord_0'
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
    
    def _create_rectangle_stroke_geometry(self, x1, y1, x2, y2, thickness):
        """Create thick stroke geometry for rectangle with mitered corners
        
        Creates an outline ring by building outer and inner rectangles
        """
        half_thick = thickness / 2
        
        # Outer rectangle (expanded by half thickness)
        outer_x1 = x1 - half_thick
        outer_y1 = y1 - half_thick
        outer_x2 = x2 + half_thick
        outer_y2 = y2 + half_thick
        
        # Inner rectangle (shrunk by half thickness)
        inner_x1 = x1 + half_thick
        inner_y1 = y1 + half_thick
        inner_x2 = x2 - half_thick
        inner_y2 = y2 - half_thick
        
        # Build the stroke as quads around the perimeter
        # Each edge is 2 triangles forming a quad
        
        vertices = []
        
        # Top edge quad
        vertices.extend([
            outer_x1, outer_y1,  # Outer top-left
            outer_x2, outer_y1,  # Outer top-right
            inner_x2, inner_y1,  # Inner top-right
            
            outer_x1, outer_y1,  # Outer top-left
            inner_x2, inner_y1,  # Inner top-right
            inner_x1, inner_y1,  # Inner top-left
        ])
        
        # Right edge quad
        vertices.extend([
            outer_x2, outer_y1,  # Outer top-right
            outer_x2, outer_y2,  # Outer bottom-right
            inner_x2, inner_y2,  # Inner bottom-right
            
            outer_x2, outer_y1,  # Outer top-right
            inner_x2, inner_y2,  # Inner bottom-right
            inner_x2, inner_y1,  # Inner top-right
        ])
        
        # Bottom edge quad
        vertices.extend([
            outer_x2, outer_y2,  # Outer bottom-right
            outer_x1, outer_y2,  # Outer bottom-left
            inner_x1, inner_y2,  # Inner bottom-left
            
            outer_x2, outer_y2,  # Outer bottom-right
            inner_x1, inner_y2,  # Inner bottom-left
            inner_x2, inner_y2,  # Inner bottom-right
        ])
        
        # Left edge quad
        vertices.extend([
            outer_x1, outer_y2,  # Outer bottom-left
            outer_x1, outer_y1,  # Outer top-left
            inner_x1, inner_y1,  # Inner top-left
            
            outer_x1, outer_y2,  # Outer bottom-left
            inner_x1, inner_y1,  # Inner top-left
            inner_x1, inner_y2,  # Inner bottom-left
        ])
        
        return np.array(vertices, dtype='f4')


    def rectangle(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a rectangle"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        if self.enable_batching:
            # Fill vertices (always same)
            fill_verts = np.array([
                x1, y1,
                x2, y1,
                x2, y2,
                x1, y1,
                x2, y2,
                x1, y2
            ], dtype='f4')
            
            # Stroke vertices - thick or thin
            if self.use_stroke and self._stroke_weight > 1.0:
                # Thick stroke - use geometry
                stroke_verts = self._create_rectangle_stroke_geometry(
                    x1, y1, x2, y2, self._stroke_weight
                )
                stroke_as_fill = True  # Render stroke as filled triangles
            else:
                # Thin stroke - use lines
                stroke_verts = np.array([
                    x1, y1, x2, y1,
                    x2, y1, x2, y2,
                    x2, y2, x1, y2,
                    x1, y2, x1, y1,
                ], dtype='f4')
                stroke_as_fill = False
            
            cmd = DrawCommand(
                type=DrawCommandType.RECTANGLE,
                fill_vertices=fill_verts,
                stroke_vertices=stroke_verts,
                color=self.fill_color if self.use_fill else None,
                use_fill=self.use_fill,
                use_stroke=self.use_stroke,
                stroke_weight=self._stroke_weight,
                stroke_color=self.stroke_color if self.use_stroke else None,
                transform=self.transform.matrix,
                layer_id=self.active_layer,
                draw_order=self.draw_order_counter,
                stroke_as_geometry=stroke_as_fill  # ← Add this flag
            )
            
            self.draw_order_counter += 1
            self.draw_queue.append(cmd)
    
    def _get_cached_unit_circle(self, segments):
        """Get cached unit circle fill (radius=1, center=0,0)"""
        if segments not in self._circle_geometry_cache:
            # Generate unit circle once
            fill_verts = []
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * (i + 1) / segments
                
                x1 = np.cos(angle1)
                y1 = np.sin(angle1)
                x2 = np.cos(angle2)
                y2 = np.sin(angle2)
                
                fill_verts.extend([0, 0, x1, y1, x2, y2])
            
            self._circle_geometry_cache[segments] = np.array(fill_verts, dtype='f4')
        
        return self._circle_geometry_cache[segments]

    def _get_cached_unit_circle_stroke(self, segments, thickness_ratio):
        """Get cached unit circle stroke (radius=1, thickness as ratio of radius)
        
        Args:
            segments: number of segments
            thickness_ratio: thickness relative to radius (e.g., 0.2 means 20% of radius)
        """
        cache_key = (segments, round(thickness_ratio, 3))  # Round to avoid float precision issues
        
        if cache_key not in self._circle_stroke_cache:
            half_thick = thickness_ratio / 2
            
            # Outer and inner radii (relative to unit circle)
            outer_radius = 1.0 + half_thick
            inner_radius = 1.0 - half_thick
            
            vertices = []
            
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * (i + 1) / segments
                
                # Outer circle points
                outer_x1 = outer_radius * np.cos(angle1)
                outer_y1 = outer_radius * np.sin(angle1)
                outer_x2 = outer_radius * np.cos(angle2)
                outer_y2 = outer_radius * np.sin(angle2)
                
                # Inner circle points
                inner_x1 = inner_radius * np.cos(angle1)
                inner_y1 = inner_radius * np.sin(angle1)
                inner_x2 = inner_radius * np.cos(angle2)
                inner_y2 = inner_radius * np.sin(angle2)
                
                # Triangle 1
                vertices.extend([
                    outer_x1, outer_y1,
                    outer_x2, outer_y2,
                    inner_x2, inner_y2,
                ])
                
                # Triangle 2
                vertices.extend([
                    outer_x1, outer_y1,
                    inner_x2, inner_y2,
                    inner_x1, inner_y1,
                ])
            
            self._circle_stroke_cache[cache_key] = np.array(vertices, dtype='f4')
        
        return self._circle_stroke_cache[cache_key]

    def _get_unit_circle_vbo(self, segments):
        """Get cached unit circle VBO"""
        if segments not in self._unit_circle_vbo_cache:
            unit_circle = self._get_cached_unit_circle(segments)
            self._unit_circle_vbo_cache[segments] = self.ctx.buffer(unit_circle)
        return self._unit_circle_vbo_cache[segments]
    
    def _get_unit_circle_stroke_vbo(self, segments, thickness_ratio):
        """Get cached unit circle stroke VBO (as ring/donut)"""
        cache_key = (segments, round(thickness_ratio, 3))
        
        if cache_key not in self._unit_circle_stroke_vbo_cache:
            unit_stroke = self._get_cached_unit_circle_stroke(segments, thickness_ratio)
            self._unit_circle_stroke_vbo_cache[cache_key] = self.ctx.buffer(unit_stroke)
    
        return self._unit_circle_stroke_vbo_cache[cache_key]
    
    def circle(self, center: Tuple[float, float], radius: float, annotate: bool = False):
        """Draw a circle (queued for batching with instancing)"""
        if self.enable_batching:
             # Transform the center point by current transform matrix
            transformed_center = self.transform.matrix * glm.vec4(center[0], center[1], 0.0, 1.0)
            world_center = (transformed_center.x, transformed_center.y)
            
            # Transform could also scale the radius
            # Extract scale from transform matrix (assuming uniform scale)
            scale = glm.length(glm.vec3(self.transform.matrix[0]))
            world_radius = radius * scale
            cmd = DrawCommand(
                type=DrawCommandType.CIRCLE,
                center=world_center,
                radius=world_radius,
                color=self.fill_color if self.use_fill else None,
                use_fill=self.use_fill,
                use_stroke=self.use_stroke,
                stroke_weight=self._stroke_weight,
                stroke_color=self.stroke_color if self.use_stroke else None,
                transform=glm.mat4(),
                layer_id=self.active_layer,
                draw_order=self.draw_order_counter,
            )
            self.draw_order_counter += 1
            self.draw_queue.append(cmd)
            
    def _create_thick_line_geometry(self, x1, y1, x2, y2, thickness):
        """Create rectangle geometry for a thick line
        
        Returns vertices for 2 triangles forming a thick line
        """
        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 0.001:
            # Degenerate line - return empty
            return np.array([], dtype='f4')
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Perpendicular vector (for thickness)
        px = -dy * thickness / 2
        py = dx * thickness / 2
        
        # Four corners of the rectangle
        vertices = np.array([
            x1 - px, y1 - py,  # Bottom-left
            x1 + px, y1 + py,  # Top-left
            x2 + px, y2 + py,  # Top-right
            
            x1 - px, y1 - py,  # Bottom-left
            x2 + px, y2 + py,  # Top-right
            x2 - px, y2 - py,  # Bottom-right
        ], dtype='f4')
        
        return vertices


    def line(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a line (batched)"""
        if self.enable_batching:
            x1, y1 = pos1
            x2, y2 = pos2
            
            if self._stroke_weight > 1.0:
                # Thick line - draw as geometry
                line_verts = self._create_thick_line_geometry(x1, y1, x2, y2, self._stroke_weight)
                cmd = DrawCommand(
                    type=DrawCommandType.LINE,
                    fill_vertices=line_verts,  # Draw as filled rectangle
                    stroke_vertices=np.array([], dtype='f4'),  # No stroke
                    color=self.stroke_color,  # Line uses stroke color as fill
                    use_fill=True,
                    use_stroke=False,
                    stroke_weight=1.0,
                    stroke_color=None,
                    transform=self.transform.matrix,
                    layer_id=self.active_layer,
                    draw_order=self.draw_order_counter
                )
            else:
                # Thin line (weight=1) - use regular line rendering
                vertices = np.array([x1, y1, x2, y2], dtype='f4')
                cmd = DrawCommand(
                    type=DrawCommandType.LINE,
                    fill_vertices=np.array([], dtype='f4'),
                    stroke_vertices=vertices,
                    color=None,
                    use_fill=False,
                    use_stroke=True,
                    stroke_weight=1.0,  # Always 1 for OpenGL lines
                    stroke_color=self.stroke_color,
                    transform=self.transform.matrix,
                    layer_id=self.active_layer,
                    draw_order=self.draw_order_counter
                )
            
            self.draw_order_counter += 1
            self.draw_queue.append(cmd)
    
    def flush_batch(self):
        """Execute all queued draw commands in batches"""
        if not self.draw_queue:
            return
        
        # Group commands by type and rendering state for efficient batching
        # This is the key optimization!
        batches = self._group_commands_into_batches()
        
        # Render each batch
        for batch in batches:
            self._render_batch(batch)
        
        # Clear the queue
        self.draw_queue.clear()
    
    def _group_commands_into_batches(self):
        """Group commands by state - transforms can differ for instanced 3D"""
        if not self.draw_queue:
            return []
        
        sorted_cmds = sorted(self.draw_queue, key=lambda c: c.draw_order)
        batches = []
        current_batch = [sorted_cmds[0]]
        
        def state_matches(cmd1, cmd2):
            """Check if two commands have matching render state"""
            # Type must match
            if cmd1.type != cmd2.type:
                return False
            
            # For 3D instanced types, transforms DON'T need to match
            if cmd1.type in [DrawCommandType.SPHERE,
                            DrawCommandType.BOX]:
                # These use instancing - different transforms are OK
                # Only check geometry/texture compatibility
                
                if cmd1.type == DrawCommandType.BOX:
                    # Same box dimensions?
                    if cmd1.fill_vertices.shape != cmd2.fill_vertices.shape:
                        return False
                    # Same texture?
                    if getattr(cmd1, 'texture_layer', None) != getattr(cmd2, 'texture_layer', None):
                        return False
                
                # Spheres always batch together
                # Colors can differ (handled per-instance)
                return True
            # For lines (concatenated, not instanced), check stroke state
            elif cmd1.type in [DrawCommandType.LINE_3D, DrawCommandType.THICK_LINE_3D, DrawCommandType.POLYLINE_3D]:
                if cmd1.use_stroke != cmd2.use_stroke:
                    return False
                if cmd1.use_stroke and cmd1.stroke_color != cmd2.stroke_color:
                    return False
                if abs(cmd1.stroke_weight - cmd2.stroke_weight) > 0.01:
                    return False
                # Lines can have different transforms (vertices are in world space already)
                return self._transforms_equal(cmd1.transform, cmd2.transform)
            else:
                if cmd1.type == DrawCommandType.CIRCLE and cmd2.type == DrawCommandType.CIRCLE:
                    # Fill state can differ per instance (different colors OK)
                    # Stroke state must match for grouping
                    if cmd1.use_stroke != cmd2.use_stroke:
                        return False
                # For 2D and non-instanced types, keep old logic
                # Fill state must match
                if cmd1.use_fill != cmd2.use_fill:
                    return False
                if cmd1.use_fill and cmd1.color != cmd2.color:
                    return False
                
                # Stroke state must match
                if cmd1.use_stroke != cmd2.use_stroke:
                    return False
                if cmd1.use_stroke:
                    if cmd1.stroke_color != cmd2.stroke_color:
                        return False
                    if abs(cmd1.stroke_weight - cmd2.stroke_weight) > 0.01:
                        return False
            
                return self._transforms_equal(cmd1.transform, cmd2.transform)
        
        prev_cmd = sorted_cmds[0]
        
        for cmd in sorted_cmds[1:]:
            if state_matches(prev_cmd, cmd):
                current_batch.append(cmd)
            else:
                # print(f"new batch size {len(current_batch)}")
                batches.append(current_batch)
                current_batch = [cmd]
                prev_cmd = cmd
        
        if current_batch:
            # print(f"new batch size {len(current_batch)}")
            batches.append(current_batch)
        
        return batches

    def _transforms_equal(self, t1, t2, epsilon=1e-5):
        """Check if two transform matrices are equal"""
        # Convert to numpy arrays
        if isinstance(t1, glm.mat4):
            a1 = np.array([t1[i][j] for i in range(4) for j in range(4)])
        else:
            a1 = np.array(t1).flatten()
        
        if isinstance(t2, glm.mat4):
            a2 = np.array([t2[i][j] for i in range(4) for j in range(4)])
        else:
            a2 = np.array(t2).flatten()
        
        return np.allclose(a1, a2, atol=epsilon)

    def _render_batch(self, commands: List[DrawCommand]):
        """Render batch with thick stroke support"""
        if not commands:
            return
        
        first_cmd = commands[0]
        if not first_cmd.is_3d:
            self._render_2d_batch(commands)
        else:
            self._render_3d_batch(commands)

    def _render_3d_batch(self,commands: List[DrawCommand]):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # self.ctx.front_face = 'ccw' 
        # self.ctx.enable(moderngl.CULL_FACE)  # Enable face culling
        first_cmd = commands[0]
        if first_cmd.type == DrawCommandType.SPHERE:
            
            # Collect all instance data
            instance_data = []
            for cmd in commands:
                
                # Flatten 4x4 matrix into 16 floats + 4 colour floats
                matrix_flat = np.array(cmd.transform, dtype='f4').T.flatten()
                
                colour = np.array(self._normalize_color(cmd.color), dtype='f4')
                instance_data.append(np.concatenate([matrix_flat, colour]))
            
            instance_array = np.array(instance_data, dtype='f4')
            
            # Create instance buffer
            instance_buffer = self.ctx.buffer(instance_array)
            
            vao = self.ctx.vertex_array(
                self.shader_3d_instanced,
                [
                    (self.sphere_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                    (instance_buffer, '16f 4f /i', 'instance_model', 'instance_color'),
                ],
                index_buffer=None  # No index buffer
            )

            # Set shared uniforms
            self.shader_3d_instanced['view'].write(self.camera.get_view_matrix())
            self.shader_3d_instanced['projection'].write(self.camera.get_projection_matrix())
            self.shader_3d_instanced['light_pos'].write(glm.vec3(self.light_pos))
            self.shader_3d_instanced['camera_pos'].write(self.camera.position)
            self.shader_3d_instanced['use_lighting'] = self.use_lighting
    

            vao.render(moderngl.TRIANGLES, instances=len(commands))
            
            vao.release()
            instance_buffer.release()

        elif first_cmd.type == DrawCommandType.BOX:
            # Similar to mesh batching
            box_vertices = first_cmd.fill_vertices
            box_vbo = self.ctx.buffer(box_vertices)
            
            
            
            # Collect instance data
            instance_data = []
            for cmd in commands:
                matrix_flat = np.array(cmd.transform, dtype='f4').T.flatten()
                colour = np.array(self._normalize_color(cmd.color), dtype='f4')
                instance_data.append(np.concatenate([matrix_flat, colour]))
            
            instance_array = np.array(instance_data, dtype='f4')
            instance_buffer = self.ctx.buffer(instance_array)
            
            vao = self.ctx.vertex_array(
                self.shader_3d_instanced,
                [
                    (box_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                    (instance_buffer, '16f 4f /i', 'instance_model', 'instance_color'),
                ]
            )
            
            # Set uniforms (similar to sphere/mesh)
            self.shader_3d_instanced['view'].write(self.camera.get_view_matrix())
            self.shader_3d_instanced['projection'].write(self.camera.get_projection_matrix())
            self.shader_3d_instanced['light_pos'].write(glm.vec3(self.light_pos))
            self.shader_3d_instanced['camera_pos'].write(self.camera.position)
            self.shader_3d_instanced['use_lighting'] = self.use_lighting
            
            # Handle texture (only works if all boxes use same texture)
            if first_cmd.texture_layer and first_cmd.texture_layer in self.layers:
                texture = self.layers[first_cmd.texture_layer]['fbo'].color_attachments[0]
                texture.use(0)
                self.shader_3d_instanced['texture0'] = 0
                self.shader_3d_instanced['use_texture'] = True
            else:
                self.shader_3d_instanced['use_texture'] = False
            
            vao.render(moderngl.TRIANGLES, instances=len(commands))
            
            vao.release()
            instance_buffer.release()
            box_vbo.release()
    
        elif first_cmd.type == DrawCommandType.THICK_LINE_3D:
            # Concatenate all line geometry
            all_vertices = np.concatenate([cmd.stroke_vertices for cmd in commands])
            
            vbo = self.ctx.buffer(all_vertices)
            vao = self.ctx.vertex_array(
                self.shader_3d,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
            )
            
            self.shader_3d['model'].write(first_cmd.transform)
            self.shader_3d['view'].write(self.camera.get_view_matrix())
            self.shader_3d['projection'].write(self.camera.get_projection_matrix())
            self.shader_3d['color'].write(glm.vec4(*self._normalize_color(first_cmd.stroke_color)))
            self.shader_3d['use_lighting'] = self.use_lighting  # Lines can now have lighting!
            self.shader_3d['use_texture'] = False
            
            vao.render(moderngl.TRIANGLES)
            
            vao.release()
            vbo.release()
        else:
            # Build vertex data with padding
            all_vertices = []
            
            for cmd in commands:
                # cmd.stroke_vertices is [x1,y1,z1, x2,y2,z2]
                verts = cmd.stroke_vertices
                for i in range(0, len(verts), 3):
                    all_vertices.extend([
                        verts[i], verts[i+1], verts[i+2],  # position (3f)
                        0.0, 0.0, 0.0,                      # normal (3f) - dummy
                        0.0, 0.0                            # texcoord (2f) - dummy
                    ])
            
            vertices_array = np.array(all_vertices, dtype='f4')
            vbo = self.ctx.buffer(vertices_array)
            vao = self.ctx.vertex_array(
                self.shader_3d,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
            )
            
            # Set uniforms
            self.shader_3d['model'].write(first_cmd.transform) 
            self.shader_3d['view'].write(self.camera.get_view_matrix())
            self.shader_3d['projection'].write(self.camera.get_projection_matrix())
            self.shader_3d['color'].write(glm.vec4(*self._normalize_color(first_cmd.stroke_color)))
            self.shader_3d['use_lighting'] = False
            self.shader_3d['use_texture'] = False
            
            vao.render(moderngl.LINES if first_cmd.type == DrawCommandType.LINE_3D else moderngl.LINE_STRIP)
            
            vao.release()
            vbo.release()

    def _render_2d_batch(self,commands: List[DrawCommand]):
        first_cmd = commands[0]
                # Set up rendering state
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # FILL PASS
        if first_cmd.type == DrawCommandType.CIRCLE:
            # Determine segments based on max radius
            max_radius = max(cmd.radius for cmd in commands if cmd.radius)
            if max_radius < 2:
                segments = 8
            elif max_radius < 10:
                segments = 16
            elif max_radius < 50:
                segments = 32
            else:
                segments = 48
            
            # FILL RENDERING
            if first_cmd.use_fill:
                unit_circle_vbo = self._get_unit_circle_vbo(segments)
                
                # Build instance data for fills
                fill_instance_data = []
                for cmd in commands:
                    if cmd.use_fill:
                        color = np.array(self._normalize_color(cmd.color), dtype='f4')
                        fill_instance_data.extend([
                            cmd.center[0], cmd.center[1],  # center (2f)
                            cmd.radius,                     # radius (1f)
                            *color                          # color (4f)
                        ])
                
                if fill_instance_data:
                    fill_instance_array = np.array(fill_instance_data, dtype='f4')
                    fill_instance_buffer = self.ctx.buffer(fill_instance_array)
                    
                    fill_vao = self.ctx.vertex_array(
                        self.shader_2d_instanced,
                        [
                            (unit_circle_vbo, '2f', 'in_position'),
                            (fill_instance_buffer, '2f 1f 4f /i', 'instance_center', 'instance_radius', 'instance_color'),
                        ]
                    )
                    self.shader_2d_instanced['projection'].write(self.camera.get_projection_matrix())
                    self.shader_2d_instanced['model'].write(glm.mat4())

                    # In _render_2d_batch for circles, right before rendering:

                    fill_vao.render(moderngl.TRIANGLES, instances=len(fill_instance_data) // 7)
                    
                    fill_vao.release()
                    fill_instance_buffer.release()
            
            # STROKE RENDERING
            if first_cmd.use_stroke:
                # Group by stroke weight (thickness ratio must match for same geometry)
                strokes_by_weight = {}
                for cmd in commands:
                    if cmd.use_stroke:
                        weight = cmd.stroke_weight
                        if weight not in strokes_by_weight:
                            strokes_by_weight[weight] = []
                        strokes_by_weight[weight].append(cmd)
                
                # Render each stroke weight group
                for stroke_weight, stroke_cmds in strokes_by_weight.items():
                    if stroke_weight > 1.0:
                        # Thick stroke - use ring geometry
                        # Use average radius for thickness ratio (or max)
                        avg_radius = sum(cmd.radius for cmd in stroke_cmds) / len(stroke_cmds)
                        thickness_ratio = stroke_weight / avg_radius
                        
                        unit_stroke_vbo = self._get_unit_circle_stroke_vbo(segments, thickness_ratio)
                        
                        # Build instance data for strokes
                        stroke_instance_data = []
                        for cmd in stroke_cmds:
                            color = np.array(self._normalize_color(cmd.stroke_color), dtype='f4')
                            stroke_instance_data.extend([
                                cmd.center[0], cmd.center[1],  # center (2f)
                                cmd.radius,                     # radius (1f)
                                *color                          # color (4f)
                            ])
                        
                        stroke_instance_array = np.array(stroke_instance_data, dtype='f4')
                        stroke_instance_buffer = self.ctx.buffer(stroke_instance_array)
                        
                        stroke_vao = self.ctx.vertex_array(
                            self.shader_2d_instanced,
                            [
                                (unit_stroke_vbo, '2f', 'in_position'),
                                (stroke_instance_buffer, '2f 1f 4f /i', 'instance_center', 'instance_radius', 'instance_color'),
                            ]
                        )
                        self.shader_2d_instanced['projection'].write(self.camera.get_projection_matrix())
                        self.shader_2d_instanced['model'].write(glm.mat4())
                        stroke_vao.render(moderngl.TRIANGLES, instances=len(stroke_instance_data) // 7)
                        
                        stroke_vao.release()
                        stroke_instance_buffer.release()
                    else:
                        # Thin stroke - use line loop (or implement line instancing)
                        # For now, fall back to old method or skip
                        pass
        elif first_cmd.use_fill and first_cmd.color is not None:
            self.shader_2d['projection'].write(self.camera.get_projection_matrix())
            self.shader_2d['model'].write(first_cmd.transform)
            try:
                all_fill_verts = np.concatenate([cmd.fill_vertices for cmd in commands 
                                                if len(cmd.fill_vertices) > 0])
            except:
                print("error concatenating verts")
                return
            
            if len(all_fill_verts) > 0:
                fill_vbo = self.ctx.buffer(all_fill_verts)
                fill_vao = self.ctx.simple_vertex_array(self.shader_2d, fill_vbo, 'in_position')
                # print(f"rendering {len(all_fill_verts)} verts")
                self.shader_2d['color'].write(glm.vec4(*self._normalize_color(first_cmd.color)))
                fill_vao.render(moderngl.TRIANGLES)
                
                fill_vao.release()
                fill_vbo.release()
        
        # STROKE PASS
        if first_cmd.use_stroke and first_cmd.stroke_color is not None:
            if first_cmd.stroke_as_geometry:
                # Thick stroke - render as filled triangles
                try:
                    stroke_verts_list = [cmd.stroke_vertices for cmd in commands 
                                        if cmd.stroke_vertices is not None and len(cmd.stroke_vertices) > 0]
                except:
                    print("error concatenating verts")
                    return
                
                if stroke_verts_list:
                    all_stroke_verts = np.concatenate(stroke_verts_list)
                    
                    stroke_vbo = self.ctx.buffer(all_stroke_verts)
                    stroke_vao = self.ctx.simple_vertex_array(self.shader_2d, stroke_vbo, 'in_position')
                    
                    self.shader_2d['color'].write(glm.vec4(*self._normalize_color(first_cmd.stroke_color)))
                    stroke_vao.render(moderngl.TRIANGLES)  # Render as filled geometry
                    
                    stroke_vao.release()
                    stroke_vbo.release()
            else:
                # Thin stroke - render as lines
                stroke_verts_list = [cmd.stroke_vertices for cmd in commands 
                                    if cmd.stroke_vertices is not None and len(cmd.stroke_vertices) > 0]
                
                if stroke_verts_list:
                    all_stroke_verts = np.concatenate(stroke_verts_list)
                    
                    stroke_vbo = self.ctx.buffer(all_stroke_verts)
                    stroke_vao = self.ctx.simple_vertex_array(self.shader_2d, stroke_vbo, 'in_position')
                    
                    self.shader_2d['color'].write(glm.vec4(*self._normalize_color(first_cmd.stroke_color)))
                    stroke_vao.render(moderngl.LINES)
                    
                    stroke_vao.release()
                    stroke_vbo.release()
                
    def polyline(self, points, closed: bool = False):
        """Draw a polyline (connected line segments)"""
        if len(points) < 2:
            return
        
        # Create vertices
        vertices = []
        for x, y in points:
            vertices.extend([x, y])
        
        if closed:
            vertices.extend([points[0][0], points[0][1]])
        
        vertices = np.array(vertices, dtype='f4')
        
        # Enable blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Set uniforms
        self.shader_2d['projection'].write(self.camera.get_projection_matrix())
        self.shader_2d['model'].write(self.transform.matrix)
        
        if self.use_stroke:
            self.shader_2d['color'].write(glm.vec4(*self._normalize_color(self.stroke_color)))
            
            # Try thick stroke
            thick_verts = self._draw_thick_stroke(vertices, closed=closed)
            
            if thick_verts is not None:
                thick_vbo = self.ctx.buffer(thick_verts)
                thick_vao = self.ctx.simple_vertex_array(self.shader_2d, thick_vbo, 'in_position')
                thick_vao.render(moderngl.TRIANGLE_STRIP)
                thick_vao.release()
                thick_vbo.release()
            else:
                vbo = self.ctx.buffer(vertices)
                vao = self.ctx.simple_vertex_array(self.shader_2d, vbo, 'in_position')
                self.ctx.line_width = 1.0
                vao.render(moderngl.LINE_STRIP)
                vao.release()
                vbo.release()

    def _draw_thick_stroke(self, vertices, closed=False):
        """Draw thick lines as quads instead of using line_width
        
        Args:
            vertices: np.array of [x1, y1, x2, y2, ...] coordinates
            closed: If True, connect last point to first
        """
        if self._stroke_weight <= 1.0:
            # Use regular lines for weight 1
            return None  # Signal to use regular line rendering
        
        thickness = self._stroke_weight
        points = vertices.reshape(-1, 2)
        
        if len(points) < 2:
            return None
        
        # If closed, add first point at end
        if closed:
            points = np.vstack([points, points[0:1]])
        
        # Build quad strip for thick line
        quad_vertices = []
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Calculate perpendicular direction
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < 0.001:
                continue
            
            # Normalize and get perpendicular
            dx /= length
            dy /= length
            px = -dy * thickness / 2
            py = dx * thickness / 2
            
            # Add quad for this segment
            quad_vertices.extend([
                x1 + px, y1 + py,
                x1 - px, y1 - py,
                x2 + px, y2 + py,
                x2 - px, y2 - py,
            ])
        
        if not quad_vertices:
            return None
        
        return np.array(quad_vertices, dtype='f4')
                
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

    def _generate_sphere_vertices(self, radius=1.0, sectors=32, rings=32):
        """Generate sphere geometry manually"""
        vertices = []
        normals = []
        texcoords = []
        
        # Generate vertices
        for ring in range(rings + 1):
            theta = ring * np.pi / rings  # 0 to pi
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            for sector in range(sectors + 1):
                phi = sector * 2 * np.pi / sectors  # 0 to 2pi
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                
                # Position
                x = radius * sin_theta * cos_phi
                y = radius * cos_theta
                z = radius * sin_theta * sin_phi
                
                # Normal (same as position for unit sphere)
                nx = sin_theta * cos_phi
                ny = cos_theta
                nz = sin_theta * sin_phi
                
                # Texcoord
                u = sector / sectors
                v = ring / rings
                
                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
                texcoords.extend([u, v])
        
        # Generate indices and convert to triangles
        triangle_verts = []
        triangle_normals = []
        triangle_texcoords = []
        
        for ring in range(rings):
            for sector in range(sectors):
                # Four vertices of the quad
                current = ring * (sectors + 1) + sector
                next_ring = current + sectors + 1
                
                # First triangle
                for idx in [current, next_ring, current + 1]:
                    triangle_verts.extend(vertices[idx*3:idx*3+3])
                    triangle_normals.extend(normals[idx*3:idx*3+3])
                    triangle_texcoords.extend(texcoords[idx*2:idx*2+2])
                
                # Second triangle
                for idx in [current + 1, next_ring, next_ring + 1]:
                    triangle_verts.extend(vertices[idx*3:idx*3+3])
                    triangle_normals.extend(normals[idx*3:idx*3+3])
                    triangle_texcoords.extend(texcoords[idx*2:idx*2+2])
        
        return (np.array(triangle_verts, dtype='f4'),
                np.array(triangle_normals, dtype='f4'),
                np.array(triangle_texcoords, dtype='f4'))
    
    def sphere(self, radius: float = 1.0, position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D sphere
        
        Args:
            radius: Sphere radius
            position: (x, y, z) center position
        """
        
        # Apply position and scale
        self.transform.push()
        self.transform.translate(position[0], position[1], position[2])
        self.transform.scale(radius)
        
        self._draw_3d_geometry(None, DrawCommandType.SPHERE)
        
        self.transform.pop()
    
    def box(self, size: Tuple[float, float, float] = (0, 0, 0), 
            position: Tuple[float, float, float] = (0, 0, 0),
            texture_layers=None):
        """Queue a box for batched rendering"""
        
        w, h, d = size
        
        # Normalize texture_layers
        if isinstance(texture_layers, int):
            # Single texture for all faces - batchable!
            tex_dict = {face: texture_layers for face in ['front', 'back', 'right', 'left', 'top', 'bottom']}
            uniform_texture = texture_layers
        elif isinstance(texture_layers, dict):
            tex_dict = {
                'front': texture_layers.get('front'),
                'back': texture_layers.get('back'),
                'right': texture_layers.get('right'),
                'left': texture_layers.get('left'),
                'top': texture_layers.get('top'),
                'bottom': texture_layers.get('bottom')
            }
            # Check if all faces use same texture
            unique_textures = set(tex_dict.values())
            if len(unique_textures) == 1:
                uniform_texture = list(unique_textures)[0]
            else:
                # Different textures per face - can't batch efficiently
                # Fall back to immediate rendering for now
                self._draw_box_immediate(w, h, d, position, tex_dict)
                return
        else:
            # No texture
            tex_dict = {face: None for face in ['front', 'back', 'right', 'left', 'top', 'bottom']}
            uniform_texture = None
        
        faces_data = self._create_box_faces(w, h, d)
        
        # Combine all faces into one mesh
        all_vertices = np.concatenate(list(faces_data.values()))
        
        self.transform.push()
        self.transform.translate(position[0], position[1], position[2])
        
        self._draw_3d_geometry(all_vertices, DrawCommandType.BOX, texture_layer=uniform_texture)
        
        self.transform.pop()
        
    def _draw_box_immediate(self, w, h, d, position, tex_dict):
        """Render a box with per-face textures immediately (can't batch)"""
        faces_data = self._create_box_faces(w, h, d)
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.transform.push()
        self.transform.translate(position[0], position[1], position[2])
        
        for face_name, vertices in faces_data.items():
            layer_id = tex_dict[face_name]
            
            vbo = self.ctx.buffer(vertices)
            shader = self.shader_3d
            vao = self.ctx.vertex_array(
                shader,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
            )
            
            # Set uniforms
            shader['projection'].write(self.camera.get_projection_matrix())
            shader['view'].write(self.camera.get_view_matrix())
            shader['model'].write(self.transform.matrix)
            
            # Texture or solid color
            if layer_id is not None and layer_id in self.layers:
                texture = self.layers[layer_id]['fbo'].color_attachments[0]
                texture.use(0)
                shader['texture0'] = 0
                shader['use_texture'] = True
            else:
                shader['use_texture'] = False
                shader['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
            
            shader['use_lighting'] = self.use_lighting
            shader['light_pos'].write(glm.vec3(self.light_pos))
            shader['camera_pos'].write(self.camera.position)
            
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()
        
        self.transform.pop()

    def _create_box_faces(self, w, h, d):
        """Create box face geometry - returns dict of interleaved vertex data"""
        faces_data = {
            'front': np.array([
                -w, -h,  d,  0, 0, 1,  0, 0,
                w, -h,  d,  0, 0, 1,  1, 0,
                w,  h,  d,  0, 0, 1,  1, 1,
                -w, -h,  d,  0, 0, 1,  0, 0,
                w,  h,  d,  0, 0, 1,  1, 1,
                -w,  h,  d,  0, 0, 1,  0, 1,
            ], dtype='f4'),
            
            'back': np.array([
                w, -h, -d,  0, 0, -1,  0, 0,
                -w, -h, -d,  0, 0, -1,  1, 0,
                -w,  h, -d,  0, 0, -1,  1, 1,
                w, -h, -d,  0, 0, -1,  0, 0,
                -w,  h, -d,  0, 0, -1,  1, 1,
                w,  h, -d,  0, 0, -1,  0, 1,
            ], dtype='f4'),
            
            'right': np.array([
                w, -h,  d,  1, 0, 0,  0, 0,
                w, -h, -d,  1, 0, 0,  1, 0,
                w,  h, -d,  1, 0, 0,  1, 1,
                w, -h,  d,  1, 0, 0,  0, 0,
                w,  h, -d,  1, 0, 0,  1, 1,
                w,  h,  d,  1, 0, 0,  0, 1,
            ], dtype='f4'),
            
            'left': np.array([
                -w, -h, -d, -1, 0, 0,  0, 0,
                -w, -h,  d, -1, 0, 0,  1, 0,
                -w,  h,  d, -1, 0, 0,  1, 1,
                -w, -h, -d, -1, 0, 0,  0, 0,
                -w,  h,  d, -1, 0, 0,  1, 1,
                -w,  h, -d, -1, 0, 0,  0, 1,
            ], dtype='f4'),
            
            'top': np.array([
                -w,  h,  d,  0, 1, 0,  0, 0,
                w,  h,  d,  0, 1, 0,  1, 0,
                w,  h, -d,  0, 1, 0,  1, 1,
                -w,  h,  d,  0, 1, 0,  0, 0,
                w,  h, -d,  0, 1, 0,  1, 1,
                -w,  h, -d,  0, 1, 0,  0, 1,
            ], dtype='f4'),
            
            'bottom': np.array([
                -w, -h, -d,  0, -1, 0,  0, 0,
                w, -h, -d,  0, -1, 0,  1, 0,
                w, -h,  d,  0, -1, 0,  1, 1,
                -w, -h, -d,  0, -1, 0,  0, 0,
                w, -h,  d,  0, -1, 0,  1, 1,
                -w, -h,  d,  0, -1, 0,  0, 1,
            ], dtype='f4'),
        }
        
        return faces_data

    def _create_3d_line_geometry(self, pos1, pos2, thickness=0.05):
        """Create a rectangular tube between two points
        
        Returns vertices for a box oriented along the line direction
        """
        p1 = np.array(pos1, dtype='f4')
        p2 = np.array(pos2, dtype='f4')
        
        # Direction vector
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 0.0001:
            return np.array([], dtype='f4')
        
        direction = direction / length
        
        # Find perpendicular vectors to create the rectangle cross-section
        # Use up vector (0,1,0) unless line is vertical
        if abs(direction[1]) > 0.99:
            up = np.array([1.0, 0.0, 0.0], dtype='f4')
        else:
            up = np.array([0.0, 1.0, 0.0], dtype='f4')
        
        # Cross products to get perpendicular axes
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right) * (thickness / 2)
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up) * (thickness / 2)
        
        # 8 corners of the rectangular tube
        corners = [
            p1 - right - up,  # 0
            p1 + right - up,  # 1
            p1 + right + up,  # 2
            p1 - right + up,  # 3
            p2 - right - up,  # 4
            p2 + right - up,  # 5
            p2 + right + up,  # 6
            p2 - right + up,  # 7
        ]
        
        # Create triangles for 6 faces (2 triangles per face)
        # Format: position (3f), normal (3f), texcoord (2f)
        vertices = []
        
        # Helper to add a quad (2 triangles)
        def add_quad(i1, i2, i3, i4, normal):
            for idx in [i1, i2, i3, i1, i3, i4]:
                vertices.extend(corners[idx])     # position
                vertices.extend(normal)            # normal
                vertices.extend([0.0, 0.0])       # texcoord (dummy)
        
        # Front face (at p1)
        add_quad(0, 1, 2, 3, -direction)
        # Back face (at p2)
        add_quad(5, 4, 7, 6, direction)
        # Four side faces
        add_quad(0, 4, 5, 1, -up)           # bottom
        add_quad(2, 6, 7, 3, up)            # top
        add_quad(1, 5, 6, 2, right)         # right
        add_quad(4, 0, 3, 7, -right)        # left
        
        return np.array(vertices, dtype='f4')

    def _create_3d_polyline_geometry(self, points, thickness=0.05, closed=False):
        """Create geometry for a polyline"""
        if len(points) < 2:
            return np.array([], dtype='f4')
        
        all_vertices = []
        
        # Create geometry for each segment
        for i in range(len(points) - 1):
            segment_verts = self._create_3d_line_geometry(points[i], points[i+1], thickness)
            if len(segment_verts) > 0:
                all_vertices.append(segment_verts)
        
        # Close the loop if needed
        if closed and len(points) > 2:
            segment_verts = self._create_3d_line_geometry(points[-1], points[0], thickness)
            if len(segment_verts) > 0:
                all_vertices.append(segment_verts)
        
        if len(all_vertices) == 0:
            return np.array([], dtype='f4')
        
        return np.concatenate(all_vertices)

    def line_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]):
        """Draw a line in 3D space
        
        Args:
            pos1: (x, y, z) start point
            pos2: (x, y, z) end point
        """

        if self._stroke_weight == 1:

            vertices = np.array([
                pos1[0], pos1[1], pos1[2],
                pos2[0], pos2[1], pos2[2]
            ], dtype='f4')

            self._draw_3d_geometry(vertices, DrawCommandType.LINE_3D)  # Sets uniforms
        else:
            thickness = self._stroke_weight * 0.01  # Scale stroke weight appropriately
            vertices = self._create_3d_line_geometry(pos1, pos2, thickness)
            
            if len(vertices) > 0:
                self._draw_3d_geometry(vertices, DrawCommandType.THICK_LINE_3D)


    def polyline_3d(self, points, closed: bool = False):
        """Draw a 3D polyline (connected line segments)
        
        Args:
            points: List of (x, y, z) coordinates
            closed: If True, connect last point back to first
        """
        if len(points) < 2:
            return
        
        if self._stroke_weight == 1:
        
            # Create vertices
            vertices = []
            for x, y, z in points:
                vertices.extend([x, y, z])
            
            if closed:
                vertices.extend([points[0][0], points[0][1], points[0][2]])
            
            vertices = np.array(vertices, dtype='f4')

            
            # Use shared 3D geometry setup
            self._draw_3d_geometry(vertices,DrawCommandType.POLYLINE_3D)
        
        else:

            thickness = self._stroke_weight * 0.01
            vertices = self._create_3d_polyline_geometry(points, thickness, closed)
            
            if len(vertices) > 0:
                self._draw_3d_geometry(vertices, DrawCommandType.THICK_LINE_3D)


    def load_obj(self, filepath):
        """Load a Wavefront OBJ file
        
        Args:
            filepath: Path to .obj file
        
        Returns:
            Dictionary with vertex data that can be passed to draw_mesh()
        """
        vertices = []
        normals = []
        texcoords = []
        
        final_vertices = []
        final_normals = []
        final_texcoords = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # Vertex positions
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                # Texture coordinates
                elif parts[0] == 'vt':
                    texcoords.append([float(parts[1]), float(parts[2])])
                
                # Normals
                elif parts[0] == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                # Faces
                elif parts[0] == 'f':
                    # Parse face indices (can be v, v/vt, v/vt/vn, or v//vn)
                    face_vertices = []
                    for i in range(1, len(parts)):
                        indices = parts[i].split('/')
                        v_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                        
                        # Get vertex position
                        v = vertices[v_idx]
                        
                        # Get texture coordinate if present
                        if len(indices) > 1 and indices[1]:
                            vt_idx = int(indices[1]) - 1
                            vt = texcoords[vt_idx] if vt_idx < len(texcoords) else [0, 0]
                        else:
                            vt = [0, 0]
                        
                        # Get normal if present
                        if len(indices) > 2 and indices[2]:
                            vn_idx = int(indices[2]) - 1
                            vn = normals[vn_idx] if vn_idx < len(normals) else [0, 1, 0]
                        else:
                            vn = [0, 1, 0]
                        
                        face_vertices.append((v, vn, vt))
                    
                    # Triangulate face (assumes convex polygon)
                    for i in range(1, len(face_vertices) - 1):
                        for vertex_data in [face_vertices[0], face_vertices[i], face_vertices[i + 1]]:
                            v, vn, vt = vertex_data
                            final_vertices.extend(v)
                            final_normals.extend(vn)
                            final_texcoords.extend(vt)
        
        # Combine into interleaved vertex data: x,y,z, nx,ny,nz, u,v
        mesh_data = []
        for i in range(len(final_vertices) // 3):
            mesh_data.extend(final_vertices[i*3:i*3+3])      # position
            mesh_data.extend(final_normals[i*3:i*3+3])       # normal
            mesh_data.extend(final_texcoords[i*2:i*2+2])     # texcoord
        
        return {
            'vertices': np.array(mesh_data, dtype='f4'),
            'vertex_count': len(final_vertices) // 3
        }

    def draw_mesh(self, mesh_data, texture_layer=None):
        """Draw a loaded mesh"""
        vertices = mesh_data['vertices']
        vertex_count = mesh_data['vertex_count']
        
        vbo = self.ctx.buffer(vertices)
        
        # Always use textured shader for OBJ meshes (they have UVs)
        shader = self.shader_3d
        
        vao = self.ctx.vertex_array(
            shader,
            [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
        )
        
        # Enable blending and depth
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Set uniforms
        shader['projection'].write(self.camera.get_projection_matrix())
        shader['view'].write(self.camera.get_view_matrix())
        shader['model'].write(self.transform.matrix)
        
        # Texture or solid color
        if texture_layer is not None and texture_layer in self.layers:
            texture = self.layers[texture_layer]['fbo'].color_attachments[0]
            texture.use(0)
            shader['texture0'] = 0
            shader['use_texture'] = True
        else:
            shader['use_texture'] = False
            shader['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
        
        # Lighting
        shader['camera_pos'] = tuple(self.camera.position)
        shader['light_pos'].write(glm.vec3(self.light_pos))
        
        # Render
        vao.render(moderngl.TRIANGLES)
        
        vao.release()
        vbo.release()
        
    def _draw_3d_geometry(self, geom, type, texture_layer=None):

        filled = type==DrawCommandType.SPHERE or type==DrawCommandType.BOX

        cmd = DrawCommand(
                type=type,
                fill_vertices=geom if filled else None,
                stroke_vertices=None if filled else geom,
                color=self.fill_color if self.use_fill else None,
                use_fill=self.use_fill,
                use_stroke=self.use_stroke,
                stroke_weight=self._stroke_weight,
                stroke_color=self.stroke_color if self.use_stroke else None,
                transform=self.transform.matrix,
                layer_id=self.active_layer,
                draw_order=self.draw_order_counter,
                is_3d=True,
                texture_layer=texture_layer
            )
            
        self.draw_order_counter += 1
        self.draw_queue.append(cmd)        
    
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


    def text(self, text_str, position, font_size=24, font='HERSHEY_SIMPLEX', 
         align='left', color=None):
        """Draw text at position
        
        Args:
            text_str: Text to draw
            position: (x, y) position
            font_size: Font size in pixels
            font: OpenCV font constant name (string) or int
            align: 'left', 'center', 'right'
            color: Text color (uses current fill color if None)
        """
        if color is None:
            color = self.fill_color
        
        # Map font names to OpenCV constants
        font_map = {
            'HERSHEY_SIMPLEX': cv2.FONT_HERSHEY_SIMPLEX,
            'HERSHEY_PLAIN': cv2.FONT_HERSHEY_PLAIN,
            'HERSHEY_DUPLEX': cv2.FONT_HERSHEY_DUPLEX,
            'HERSHEY_COMPLEX': cv2.FONT_HERSHEY_COMPLEX,
            'HERSHEY_TRIPLEX': cv2.FONT_HERSHEY_TRIPLEX,
            'HERSHEY_COMPLEX_SMALL': cv2.FONT_HERSHEY_COMPLEX_SMALL,
            'HERSHEY_SCRIPT_SIMPLEX': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            'HERSHEY_SCRIPT_COMPLEX': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        }
        
        if isinstance(font, str):
            font = font_map.get(font, cv2.FONT_HERSHEY_SIMPLEX)
        
        # Calculate scale from font_size
        font_scale = font_size / 30.0  # Rough conversion
        thickness = max(1, int(font_size / 15))
        
        # Get text size for alignment
        (text_width, text_height), baseline = cv2.getTextSize(
            text_str, font, font_scale, thickness
        )
        
        x, y = position
        
        # Adjust position based on alignment
        if align == 'center':
            x = x - text_width // 2
        elif align == 'right':
            x = x - text_width
        # 'left' needs no adjustment
        
        # Adjust y to be baseline (OpenCV draws from baseline)
        y = y + text_height
        
        # Get current framebuffer
        if self.active_layer is not None:
            fbo = self.layers[self.active_layer]['fbo']
        else:
            fbo = self.ctx.screen
        
        # Get texture format
        texture = fbo.color_attachments[0]
        components = texture.components  # Should be 4 for RGBA
        
        # Read pixels with correct number of components
        pixels = fbo.read(components=components)
        img = np.frombuffer(pixels, dtype=np.uint8)
        img = img.reshape((self.height, self.width, components))
        img = np.flipud(img)  # Flip to correct orientation
        
        # Convert to BGR for OpenCV (handle RGBA or RGB)
        if components == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Prepare color
        if len(color) == 4:
            # RGBA color
            color_bgr = (
                int(color[2] * 255) if color[2] <= 1 else int(color[2]),
                int(color[1] * 255) if color[1] <= 1 else int(color[1]),
                int(color[0] * 255) if color[0] <= 1 else int(color[0])
            )
        else:
            # RGB color
            color_bgr = (
                int(color[2] * 255) if color[2] <= 1 else int(color[2]),
                int(color[1] * 255) if color[1] <= 1 else int(color[1]),
                int(color[0] * 255) if color[0] <= 1 else int(color[0])
            )
        
        # Draw text on image
        cv2.putText(img, text_str, (int(x), int(y)), font, font_scale,
                    color_bgr, thickness, cv2.LINE_AA)
        
        # Convert back to original format
        if components == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = np.flipud(img)  # Flip back
        
        # Write back to texture with correct size
        texture.write(img.tobytes())

    def text_size(self, text_str, font_size=24, font='HERSHEY_SIMPLEX'):
        """Get the size of text without rendering it
        
        Args:
            text_str: Text to measure
            font_size: Font size in pixels
            font: OpenCV font constant name
        
        Returns:
            (width, height) tuple
        """
        font_map = {
            'HERSHEY_SIMPLEX': cv2.FONT_HERSHEY_SIMPLEX,
            'HERSHEY_PLAIN': cv2.FONT_HERSHEY_PLAIN,
            'HERSHEY_DUPLEX': cv2.FONT_HERSHEY_DUPLEX,
            'HERSHEY_COMPLEX': cv2.FONT_HERSHEY_COMPLEX,
            'HERSHEY_TRIPLEX': cv2.FONT_HERSHEY_TRIPLEX,
            'HERSHEY_COMPLEX_SMALL': cv2.FONT_HERSHEY_COMPLEX_SMALL,
            'HERSHEY_SCRIPT_SIMPLEX': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            'HERSHEY_SCRIPT_COMPLEX': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        }
        
        if isinstance(font, str):
            font = font_map.get(font, cv2.FONT_HERSHEY_SIMPLEX)
        
        font_scale = font_size / 30.0
        thickness = max(1, int(font_size / 15))
        
        (width, height), baseline = cv2.getTextSize(
            text_str, font, font_scale, thickness
        )
        
        return (width, height + baseline)
    
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