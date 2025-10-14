"""
Dorothy - Refactored with ModernGL for 3D support and GPU acceleration

This refactored version maintains the original Processing-like API while adding:
- GPU-accelerated rendering with ModernGL
- Native 3D support (sphere, box, camera, lighting)
- Much better performance
- Backward compatibility with existing Dorothy code
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import glm
from typing import Tuple, Optional, Callable
import time
from .css_colours import css_colours
from .Audio import *
from time import sleep
import sys
import signal
import wave
import subprocess
import importlib
import traceback
import inspect
import datetime


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
        self.layer_counter = 0
        self.active_layer = None  # Currently rendering to a layer
        
    def _setup_shaders(self):
        """Initialize shader programs"""
        
        # Basic 3D shader with lighting
        self.shader_3d = self.ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                
                in vec3 in_position;
                in vec3 in_normal;
                
                out vec3 v_normal;
                out vec3 v_position;
                
                void main() {
                    vec4 world_pos = model * vec4(in_position, 1.0);
                    v_position = world_pos.xyz;
                    v_normal = mat3(transpose(inverse(model))) * in_normal;
                    gl_Position = projection * view * world_pos;
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform vec4 color;
                uniform vec3 light_pos;
                uniform vec3 camera_pos;
                uniform bool use_lighting;
                
                in vec3 v_normal;
                in vec3 v_position;
                
                out vec4 fragColor;
                
                void main() {
                    if (use_lighting) {
                        vec3 normal = normalize(v_normal);
                        vec3 light_dir = normalize(light_pos - v_position);
                        vec3 view_dir = normalize(camera_pos - v_position);
                        vec3 reflect_dir = reflect(-light_dir, normal);
                        
                        // Ambient
                        float ambient = 0.3;
                        
                        // Diffuse
                        float diffuse = max(dot(normal, light_dir), 0.0);
                        
                        // Specular
                        float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.5;
                        
                        float lighting = ambient + diffuse + specular;
                        fragColor = vec4(color.rgb * lighting, color.a);
                    } else {
                        fragColor = color;
                    }
                }
            '''
        )
        
        # Simple 2D shader
        self.shader_2d = self.ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 projection;
                uniform mat4 model;
                
                in vec2 in_position;
                
                void main() {
                    gl_Position = projection * model * vec4(in_position, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform vec4 color;
                out vec4 fragColor;
                
                void main() {
                    fragColor = color;
                }
            '''
        )
        
        # Texture shader for rendering layers
        self.shader_texture = self.ctx.program(
            vertex_shader='''
                #version 330
                
                in vec2 in_position;
                in vec2 in_texcoord;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform sampler2D texture0;
                uniform float alpha;
                
                in vec2 v_texcoord;
                out vec4 fragColor;
                
                void main() {
                    vec4 texColor = texture(texture0, v_texcoord);
                    fragColor = vec4(texColor.rgb, texColor.a * alpha);
                }
            '''
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
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.screen.use()
        
        print(f"Created layer {layer_id}: {self.width}x{self.height}")
        
        return layer_id
    
    def begin_layer(self, layer_id: int):
        """Start rendering to a specific layer
        
        Args:
            layer_id: The layer to render to (from get_layer())
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        print(f"Begin rendering to layer {layer_id}")
        self.active_layer = layer_id
        fbo = self.layers[layer_id]['fbo']
        fbo.use()
        
        # DON'T clear - we want to preserve existing content for trails
        # If you want to clear, call clear_layer() first
    
    def end_layer(self):
        """Stop rendering to layer, return to screen"""
        print(f"End layer, returning to screen")
        self.active_layer = None
        self.ctx.screen.use()
    
    def draw_layer(self, layer_id: int, alpha: float = 1.0, x: int = 0, y: int = 0):
        """Draw a layer to the screen with optional transparency
        
        Args:
            layer_id: The layer to draw
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)
            x, y: Position offset (TODO: implement positioning)
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        print(f"Drawing layer {layer_id} with alpha={alpha}")
        
        # Save which framebuffer we were rendering to
        was_rendering_to_layer = self.active_layer
        
        # Make sure we're rendering to screen
        self.ctx.screen.use()
        self.active_layer = None
        
        # Enable proper alpha blending for layer compositing
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA
        )
        
        # Bind the layer's texture
        texture = self.layers[layer_id]['texture']
        texture.use(0)
        
        # Set shader uniforms
        self.shader_texture['texture0'] = 0
        self.shader_texture['alpha'] = alpha
        
        # Draw fullscreen quad with the texture
        self.quad_vao.render(moderngl.TRIANGLES)
        
        print(f"Layer {layer_id} drawn")
        
        # Restore the framebuffer we were rendering to
        if was_rendering_to_layer is not None:
            self.active_layer = was_rendering_to_layer
            self.layers[was_rendering_to_layer]['fbo'].use()
    
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
        fbo.clear(*color)
    
    # ====== Image/Texture Pasting ======
    
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
        
        print(f"Pasting image: {w}x{h} at {position}, target size: {target_w}x{target_h}")
        
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
        
        # Create a shader program that combines 2D transform with texture
        if not hasattr(self, 'shader_texture_2d'):
            print("Creating shader_texture_2d")
            self.shader_texture_2d = self.ctx.program(
                vertex_shader='''
                    #version 330
                    
                    uniform mat4 projection;
                    uniform mat4 model;
                    
                    in vec2 in_position;
                    in vec2 in_texcoord;
                    
                    out vec2 v_texcoord;
                    
                    void main() {
                        v_texcoord = in_texcoord;
                        gl_Position = projection * model * vec4(in_position, 0.0, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 330
                    
                    uniform sampler2D texture0;
                    uniform float alpha;
                    
                    in vec2 v_texcoord;
                    out vec4 fragColor;
                    
                    void main() {
                        vec4 texColor = texture(texture0, v_texcoord);
                        fragColor = vec4(texColor.rgb, texColor.a * alpha);
                    }
                '''
            )
        
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
        
        print(f"Camera mode: {self.camera.mode}")
        print(f"Transform matrix: {self.transform.matrix}")
        
        # Bind texture and render
        texture.use(0)
        vao.render(moderngl.TRIANGLES)
        
        print("Paste rendered")
        
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
        
        # OpenGL expects origin at bottom-left, flip vertically
        img = np.flipud(img)
        
        return img
        
    def _normalize_color(self, color: Tuple) -> Tuple[float, float, float, float]:
        """Convert color from 0-255 to 0-1 range"""
        if len(color) == 3:
            return (color[0]/255, color[1]/255, color[2]/255, 1.0)
        return (color[0]/255, color[1]/255, color[2]/255, color[3]/255)
    
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
    
    # ====== 3D Drawing Methods ======
    
    def sphere(self, radius: float = 1.0, detail: int = 32):
        """Draw a 3D sphere"""
        if not self.sphere_geometry:
            self.sphere_geometry = geometry.sphere(radius=radius, sectors=detail, rings=detail)
        
        self._draw_3d_geometry(self.sphere_geometry)
    
    def box(self, width: float = 1.0, height: float = 1.0, depth: float = 1.0):
        """Draw a 3D box"""
        if not self.box_geometry:
            self.box_geometry = geometry.cube(size=(width, height, depth))
        
        self._draw_3d_geometry(self.box_geometry)
    
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
        self.fill_color = color
        self.use_fill = True
    
    def no_fill(self):
        """Disable fill"""
        self.use_fill = False
    
    def stroke(self, color: Tuple):
        """Set stroke color"""
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


class DorothyWindow(mglw.WindowConfig):
    """Internal window configuration for moderngl-window"""
    
    gl_version = (3, 3)
    title = "Dorothy - ModernGL"
    resizable = True
    cursor = True  # Enable cursor tracking
    samples = 4  # Enable MSAA for smoother lines
    vsync = True  # Enable vsync to cap framerate and reduce CPU usage
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get the Dorothy instance that's waiting for us
        self.dorothy = Dorothy._pending_instance
        self.start_time_millis = int(round(time.time() * 1000))
        # Setup renderer with the context
        self.dorothy.renderer = DorothyRenderer(
            self.ctx, 
            self.window_size[0], 
            self.window_size[1]
        )
        self.dorothy.wnd = self.wnd
        self.dorothy._initialized = True
        self.dorothy.music = Audio()
        
        # Set default 2D camera mode
        self.dorothy.renderer.camera.mode = '2d'
        
        # Try to activate/focus the window
        try:
            if hasattr(self.wnd._window, 'activate'):
                self.wnd._window.activate()
                print("Window activated")
            if hasattr(self.wnd._window, 'set_visible'):
                self.wnd._window.set_visible(True)
                print("Window set visible")
        except Exception as e:
            print(f"Could not activate window: {e}")
        
        # Call user setup
        if self.dorothy.setup_fn:
            self.dorothy.setup_fn()
    
        self._colours = {name.replace(" ", "_"): rgb for name, rgb in css_colours.items()}
        print("done load colours")

    def __getattr__(self, name):
        # Dynamically retrieve colour attributes
        try:
            return self._colours[name]
        except KeyError:
            raise AttributeError(f"{name} not found in colour attributes")
        
    def on_render(self, render_time: float, frame_time: float):
        """Called every frame"""
        frame_started_at = int(round(time.time() * 1000))
        # Signal handler function
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C! Closing the window.')
            self.dorothy.exit()

        try:
            # Link the signal handler to SIGINT
            signal.signal(signal.SIGTSTP, signal_handler)
        except Exception as e:
            done = True
            print(e)
            traceback.print_exc()
            self.dorothy.exit()  

        # Reset transforms
        self.dorothy.renderer.transform.reset()
        
        # Call user draw
        try:
            if self.dorothy.draw_fn:
                self.dorothy.draw_fn()
        except Exception as e:
            if self.dorothy.frames < 5:
                print(f"Error in draw(): {e}")
        
        self.dorothy.frames += 1
    
    def on_mouse_position_event(self, x, y, dx, dy):
        self.dorothy.mouse_x = int(x)
        self.dorothy.mouse_y = int(y)
        # print(self.dorothy.mouse_x, self.dorothy.mouse_y)
        # print("Mouse position:", x, y, dx, dy)

    def on_mouse_drag_event(self, x, y, dx, dy):
        print("Mouse drag:", x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        print("Mouse wheel:", x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))

    def on_mouse_release_event(self, x: int, y: int, button: int):
        print("Mouse button {} released at {}, {}".format(button, x, y))
    
    def key_event(self, key, action, modifiers):
        """Handle keyboard events"""
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Q or key == self.wnd.keys.ESCAPE:
                self.wnd.close()
                print("Window closing...")
    
    def resize(self, width: int, height: int):
        if self.dorothy.renderer:
            self.dorothy.renderer.width = width
            self.dorothy.renderer.height = height
            self.dorothy.renderer.camera.width = width
            self.dorothy.renderer.camera.height = height
            self.dorothy.renderer.camera.aspect = width / height


class Dorothy:
    """
    Main Dorothy class with Processing-like API
    
    Usage:
        dot = Dorothy()
        
        # Test layer transparency
    class LayerTransparencyTest:
        def __init__(self):
            self.layer1 = None
            self.layer2 = None
            dot.start_loop(self.setup, self.draw)
        
        def setup(self):
            print("=== Layer Transparency Test ===")
            self.layer1 = dot.get_layer()
            self.layer2 = dot.get_layer()
            
            # Draw static content to layer 1 (red circle)
            dot.begin_layer(self.layer1)
            dot.fill((255, 0, 0))
            dot.no_stroke()
            dot.circle((300, 300), 100)
            dot.end_layer()
            
            # Draw static content to layer 2 (blue circle)
            dot.begin_layer(self.layer2)
            dot.fill((0, 0, 255))
            dot.no_stroke()
            dot.circle((500, 300), 100)
            dot.end_layer()
            
            print("Layers created and drawn to")
        
        def draw(self):
            # Clear background to dark gray
            dot.background((50, 50, 50))
            
            # Draw both layers with different alpha values
            dot.draw_layer(self.layer1, alpha=1.0)  # Full opacity
            dot.draw_layer(self.layer2, alpha=0.5)  # Half transparent
            
            # Draw some direct shapes on screen for comparison
            dot.fill((0, 255, 0))
            dot.no_stroke()
            dot.circle((400, 150), 50)  # Green circle on screen
            
            if dot.frames == 1:
                print("You should see:")
                print("- Red circle (opaque) at left")
                print("- Blue circle (50% transparent) at right - should see gray through it")
                print("- Green circle at top")
    
    class MySketch:
            def __init__(self):
                dot.start_loop(self.setup, self.draw)
            
            def setup(self):
                dot.background((255, 255, 255))
            
            def draw(self):
                dot.fill((255, 0, 0))
                dot.circle((400, 300), 50)
        
        MySketch()
    """
    
    # Class variables for window management
    _pending_instance = None
    _instance = None
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "Dorothy"):
        Dorothy._instance = self
        Dorothy._pending_instance = self
        
        # Window configuration
        self.window_size = (width, height)
        self.window_title = title
        
        # Renderer (will be initialized when window is created)
        self.renderer = None
        self.wnd = None
        self._initialized = False
        
        # User sketch
        self.setup_fn = None
        self.draw_fn = None
        
        # Processing-like properties
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.frames = 0
        self.start_time = time.time()
        
        # Color constants (for compatibility)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
    
    def start_loop(self, setup_fn: Callable, draw_fn: Callable):
        """Start the render loop with setup and draw functions"""
        self.setup_fn = setup_fn
        self.draw_fn = draw_fn
        
        # Configure and run the window
        DorothyWindow.window_size = self.window_size
        DorothyWindow.title = self.window_title
        
        # Run the window (this will call setup_fn when ready)
        mglw.run_window_config(DorothyWindow)
    
    def start_livecode_loop(self, sketch_module):
        """Start a live coding loop that reloads code on file changes
        
        Args:
            sketch_module: The module containing MySketch class
            
        Example:
            import my_sketch
            dot.start_livecode_loop(my_sketch)
        """
        import importlib
        import traceback
        import inspect
        
        my_sketch = sketch_module.MySketch()
        self.was_error = False
        
        def setup_wrapper():
            try:
                importlib.reload(sketch_module)
                new_class = sketch_module.MySketch
                my_sketch.__class__ = new_class
                my_sketch.setup()
                self.was_error = False
            except Exception:
                if not self.was_error:
                    print("error in setup, code not updated")
                    print(traceback.format_exc())
                    self.was_error = True
        
        def draw_wrapper():
            try:
                importlib.reload(sketch_module)
                new_class = sketch_module.MySketch
                my_sketch.__class__ = new_class
                
                # Handle run_once function
                if hasattr(my_sketch, 'run_once'):
                    func_key = inspect.getsource(my_sketch.run_once)
                    if not hasattr(my_sketch, 'old_once_func'):
                        my_sketch.old_once_func = func_key
                    if my_sketch.old_once_func != func_key:
                        my_sketch.once_ran = False
                        my_sketch.old_once_func = func_key
                    if not getattr(my_sketch, 'once_ran', False):
                        my_sketch.run_once()
                        my_sketch.once_ran = True
                
                my_sketch.draw()
                self.was_error = False
            except Exception:
                if not self.was_error:
                    print(traceback.format_exc())
                    print("error in draw loop, code not updated")
                    self.was_error = True
        
        self.start_loop(setup_wrapper, draw_wrapper)    

    def exit(self):
        # self.music.clean_up()
        sys.exit(0)

    # ====== Properties ======
    
    @property
    def width(self) -> int:
        return self.window_size[0]
    
    @property
    def height(self) -> int:
        return self.window_size[1]
    
    @property
    def millis(self) -> float:
        """Time in milliseconds since start"""
        return (time.time() - self.start_time) * 1000
    
    # ====== Drawing API (delegates to renderer) ======
    
    def _ensure_renderer(self):
        """Ensure renderer is initialized"""
        if not self._initialized:
            raise RuntimeError("Dorothy not initialized. Call start_loop() first.")
    
    def background(self, color: Tuple):
        """Set background color and clear"""
        self._ensure_renderer()
        self.renderer.clear(color)
    
    def fill(self, color: Tuple):
        """Set fill color"""
        self._ensure_renderer()
        self.renderer.fill(color)
    
    def no_fill(self):
        """Disable fill"""
        self._ensure_renderer()
        self.renderer.no_fill()
    
    def stroke(self, color: Tuple):
        """Set stroke color"""
        self._ensure_renderer()
        self.renderer.stroke(color)
    
    def no_stroke(self):
        """Disable stroke"""
        self._ensure_renderer()
        self.renderer.no_stroke()
    
    def set_stroke_weight(self, weight: float):
        """Set stroke weight"""
        self._ensure_renderer()
        self.renderer.set_stroke_weight(weight)
    
    # 2D shapes
    def circle(self, center: Tuple[float, float], radius: float, annotate: bool = False):
        """Draw a circle"""
        self._ensure_renderer()
        self.renderer.circle(center, radius, annotate)
    
    def rectangle(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a rectangle"""
        self._ensure_renderer()
        self.renderer.rectangle(pos1, pos2, annotate)
    
    def line(self, pos1: Tuple[float, float], pos2: Tuple[float, float], annotate: bool = False):
        """Draw a line"""
        self._ensure_renderer()
        self.renderer.line(pos1, pos2, annotate)
    
    # 3D shapes
    def sphere(self, radius: float = 1.0):
        """Draw a 3D sphere"""
        self._ensure_renderer()
        self.renderer.sphere(radius)
    
    def box(self, width: float = 1.0, height: float = 1.0, depth: float = 1.0):
        """Draw a 3D box"""
        self._ensure_renderer()
        self.renderer.box(width, height, depth)
    
    # Transforms
    def push_matrix(self):
        """Save current transformation"""
        self._ensure_renderer()
        self.renderer.push_matrix()
    
    def pop_matrix(self):
        """Restore transformation"""
        self._ensure_renderer()
        self.renderer.pop_matrix()
    
    def translate(self, x: float, y: float, z: float = 0):
        """Translate"""
        self._ensure_renderer()
        self.renderer.translate(x, y, z)
    
    def rotate(self, angle: float, x: float = 0, y: float = 0, z: float = 1):
        """Rotate (angle in radians)"""
        self._ensure_renderer()
        self.renderer.rotate(angle, x, y, z)
    
    def scale(self, s: float):
        """Scale uniformly"""
        self._ensure_renderer()
        self.renderer.scale(s)
    
    def reset_transforms(self):
        """Reset transforms"""
        self._ensure_renderer()
        self.renderer.reset_transforms()
    
    # Camera
    def camera_3d(self):
        """Switch to 3D camera mode"""
        self._ensure_renderer()
        self.renderer.camera.mode = '3d'
    
    def camera_2d(self):
        """Switch to 2D camera mode (orthographic)"""
        self._ensure_renderer()
        self.renderer.camera.mode = '2d'
    
    def set_camera(self, eye: Tuple[float, float, float], 
                   target: Tuple[float, float, float] = (0, 0, 0)):
        """Set camera position and target"""
        self._ensure_renderer()
        self.renderer.camera.position = glm.vec3(*eye)
        self.renderer.camera.target = glm.vec3(*target)
    
    # Layers
    def get_layer(self) -> int:
        """Create a new rendering layer (offscreen framebuffer)
        
        Returns:
            layer_id: Unique identifier for this layer
            
        Example:
            layer = dot.get_layer()
            dot.begin_layer(layer)
            # Draw to layer...
            dot.circle((100, 100), 50)
            dot.end_layer()
            # Later, composite the layer
            dot.draw_layer(layer, alpha=0.5)
        """
        self._ensure_renderer()
        return self.renderer.get_layer()
    
    def begin_layer(self, layer_id: int):
        """Start rendering to a specific layer
        
        Args:
            layer_id: The layer ID from get_layer()
        """
        self._ensure_renderer()
        self.renderer.begin_layer(layer_id)
    
    def end_layer(self):
        """Stop rendering to layer, return to screen"""
        self._ensure_renderer()
        self.renderer.end_layer()
    
    def draw_layer(self, layer_id: int, alpha: float = 1.0):
        """Draw a layer to the screen with transparency
        
        Args:
            layer_id: The layer to draw
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)
        """
        self._ensure_renderer()
        self.renderer.draw_layer(layer_id, alpha)
    
    def release_layer(self, layer_id: int):
        """Free a layer's resources when no longer needed
        
        Args:
            layer_id: The layer to release
        """
        self._ensure_renderer()
        self.renderer.release_layer(layer_id)
    
    # Images
    def paste(self, image: np.ndarray, position: Tuple[int, int], 
              size: Optional[Tuple[int, int]] = None, alpha: float = 1.0):
        """Paste a numpy array (image) onto the canvas
        
        Args:
            image: NumPy array of pixels. Supports:
                   - (H, W, 3) for RGB
                   - (H, W, 4) for RGBA  
                   - (H, W) for grayscale
                   Values can be uint8 (0-255) or float (0.0-1.0)
            position: (x, y) top-left corner position
            size: Optional (width, height) to resize. None = original size
            alpha: Overall transparency (0.0-1.0)
            
        Example:
            # Load image with PIL/cv2/etc
            import cv2
            img = cv2.imread('image.png')
            
            # Paste at position
            dot.paste(img, (100, 100))
            
            # Paste with resize and transparency
            dot.paste(img, (200, 200), size=(100, 100), alpha=0.5)
        """
        self._ensure_renderer()
        self.renderer.paste(image, position, size, alpha)


# ====== Example Usage ======

if __name__ == "__main__":
    # Create Dorothy instance - now works correctly!
    dot = Dorothy(width=800, height=600, title="Dorothy Demo")
    
    # Simple test to verify basic drawing
    class SimpleTest:
        def __init__(self):
            dot.start_loop(self.setup, self.draw)
        
        def setup(self):
            print("Setup called!")
            print("Move mouse around the window to test mouse events")
            # Camera is 2D by default
        
        def draw(self):
            # Clear background
            dot.background((50, 50, 60))
            
            # Draw a red circle
            dot.fill((255, 0, 0))
            dot.no_stroke()
            dot.circle((400, 300), 100)
            
            # Draw a blue rectangle with stroke
            dot.fill((0, 0, 255))
            dot.stroke((255, 255, 0))
            dot.set_stroke_weight(3)
            dot.rectangle((100, 100), (200, 200))
            
            # Draw a line
            dot.stroke((0, 255, 0))
            dot.set_stroke_weight(5)
            dot.line((50, 50), (750, 550))
            
            # Draw circle at mouse position
            dot.fill((255, 255, 255))
            dot.no_stroke()
            dot.circle((dot.mouse_x, dot.mouse_y), 20)
            
            # Print mouse position to console (throttled by frame rate)
            if dot.frames % 30 == 0:  # Print every 30 frames
                print(f"Mouse: ({dot.mouse_x}, {dot.mouse_y}), Down: {dot.mouse_down}")
    
    class MySketch:
        def __init__(self):
            self.angle = 0
            self.trail_layer = None
            dot.start_loop(self.setup, self.draw)
        
        def setup(self):
            print("=== MySketch Setup ===")
            self.trail_layer = dot.get_layer()
            print(f"Trail layer ID: {self.trail_layer}")
        
        def draw(self):
            # Clear screen
            dot.background((30, 30, 40))
            
            # Calculate position
            x = 400 + 200 * np.cos(self.angle)
            y = 300 + 200 * np.sin(self.angle)
            
            # Draw to trail layer
            dot.begin_layer(self.trail_layer)
            
            # Optional: fade effect with semi-transparent background
            if dot.frames % 2 == 0:  # Only fade every other frame for slower fade
                dot.fill((30, 30, 40))
                dot.no_stroke()
                dot.rectangle((0, 0), (800, 600))
            
            # Draw new circle
            dot.fill((255, 100, 100))
            dot.no_stroke()
            dot.circle((x, y), 20)
            
            dot.end_layer()
            
            # Draw the trail layer to screen
            dot.draw_layer(self.trail_layer, alpha=1.0)
            
            # Also draw current position directly on screen (bright green)
            dot.fill((100, 255, 100))
            dot.circle((x, y), 10)
            
            self.angle += 0.05
            
            # Debug every 60 frames
            if dot.frames % 60 == 0:
                print(f"Frame {dot.frames}, angle={self.angle:.2f}, pos=({x:.0f}, {y:.0f})")
    
    # Alternate example with 3D:
    class Example3D:
        def __init__(self):
            self.angle = 0
            dot.start_loop(self.setup, self.draw)
        
        def setup(self):
            print("3D Setup!")
            dot.camera_3d()
            dot.set_camera((0, 0, 5), (0, 0, 0))
        
        def draw(self):
            dot.background((30, 30, 40))
            
            # Draw 3D rotating cube
            dot.push_matrix()
            dot.translate(0, 0, 0)
            dot.rotate(self.angle, 1, 1, 0)
            dot.fill((255, 100, 100))
            dot.box(1, 1, 1)
            dot.pop_matrix()
            
            # Draw 3D sphere
            dot.push_matrix()
            dot.translate(-2, 0, 0)
            dot.fill((100, 255, 100))
            dot.sphere(0.5)
            dot.pop_matrix()
            
            self.angle += 0.01
    
    # Example with image pasting:
    class ImageExample:
        def __init__(self):
            self.angle = 0
            self.image = None
            dot.start_loop(self.setup, self.draw)
        
        def setup(self):
            print("Image Example Setup!")
            dot.camera_2d()
            
            # Create a procedural image (checkerboard pattern)
            # In real use, you'd load this with cv2.imread() or PIL
            size = 100
            self.image = np.zeros((size, size, 4), dtype=np.uint8)
            
            # Create checkerboard
            square_size = 10
            for i in range(size):
                for j in range(size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        self.image[i, j] = [255, 100, 100, 255]  # Red
                    else:
                        self.image[i, j] = [100, 100, 255, 255]  # Blue
        
        def draw(self):
            dot.background((30, 30, 40))
            
            # Paste image at different positions with different effects
            
            # Static image
            dot.paste(self.image, (50, 50))
            
            # Rotated position (using angle)
            x = 400 + 150 * np.cos(self.angle)
            y = 300 + 150 * np.sin(self.angle)
            dot.paste(self.image, (int(x), int(y)), size=(50, 50), alpha=0.8)
            
            # Scaled and faded
            scale = 0.5 + 0.5 * np.sin(self.angle * 2)
            size = int(100 * scale)
            dot.paste(self.image, (600, 400), size=(size, size), alpha=scale)
            
            self.angle += 0.02
    
    # Run the sketch - window will open and render
    # Choose which example to run:
    # SimpleTest()  # Start with this to verify basic drawing works
    LayerTransparencyTest()  # Test if layers support transparency
    # MySketch()  # 2D with trails - should work now with debugging
    # Example3D()  # 3D rotating objects
    # ImageExample()  # Image pasting demonstration