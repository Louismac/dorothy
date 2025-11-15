"""Core Dorothy renderer - orchestrates all components"""
import moderngl
import glm
import numpy as np
from typing import Tuple, Optional

from .state import Transform, Camera
from .geometry_cache import GeometryCache
from .batch_manager import BatchManager
from .primitives_2d import Primitives2D
from .primitives_3d import Primitives3D
from .layers import LayerManager
from .effects import EffectsManager
from ..DorothyShaders import DOTSHADERS

class DorothyRenderer:
    """Core rendering engine - delegates to specialized components"""
    
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
        
        # Batching
        self.enable_batching = True
        self.draw_queue = []
        self.draw_order_counter = 0
        
        # Transform and camera
        self.transform = Transform()
        self.camera = Camera(width, height)
        self._identity_matrix = glm.mat4()
        
        # 3D lighting
        self.light_pos = (5, 5, 5)
        self.use_lighting = True
        self.ambient_light = 0.3
        
        # Background
        self.background_color = (0, 0, 0, 1)
        
        # Setup components
        self._setup_shaders()
        self.geometry = GeometryCache(ctx)
        self.geometry.initialize_sphere_vbo()
        self.batch_manager = BatchManager(self)
        self.primitives_2d = Primitives2D(self)
        self.primitives_3d = Primitives3D(self)
        self.layer_manager = LayerManager(self)
        self.effects = EffectsManager(self)
    
    def _setup_shaders(self):
        """Initialize all shader programs"""
        # 3D shaders
        self.shader_3d_instanced = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D_INSTANCED,
            fragment_shader=DOTSHADERS.FRAG_3D_INSTANCED
        )
        
        self.shader_3d = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D_TEXTURED,
            fragment_shader=DOTSHADERS.FRAG_3D_TEXTURED
        )

        self.shader_3d_instanced_line = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D_INSTANCED_LINE,
            fragment_shader=DOTSHADERS.FRAG_3D_INSTANCED_LINE
        )
        
        self.shader_3d_instanced_thick_line = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_3D_INSTANCED_THICK_LINE,
            fragment_shader=DOTSHADERS.FRAG_3D_INSTANCED_THICK_LINE  # Use the thick line fragment shader
        )
        
        # 2D instanced shaders
        self.shader_2d_instanced = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D_INSTANCED,
            fragment_shader=DOTSHADERS.FRAG_2D_INSTANCED
        )
        
        self.shader_2d_instanced_rect = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D_INSTANCED_RECT,
            fragment_shader=DOTSHADERS.FRAG_2D_INSTANCED
        )
        
        self.shader_2d_instanced_line = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D_INSTANCED_LINE,
            fragment_shader=DOTSHADERS.FRAG_2D_INSTANCED_LINE
        )
        
        self.shader_2d_instanced_thick_line = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D_INSTANCED_THICK_LINE,
            fragment_shader=DOTSHADERS.FRAG_2D_INSTANCED_LINE
        )
        
        # Simple 2D shader
        self.shader_2d = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_2D,
            fragment_shader=DOTSHADERS.FRAG_2D
        )
        
        # Texture shaders
        self.shader_texture = self.ctx.program(
            vertex_shader=DOTSHADERS.VERT_TEXTURE,
            fragment_shader=DOTSHADERS.FRAG_TEXTURE
        )
        
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
    
    # ===== Core Methods =====
    
    def clear(self, color: Optional[Tuple] = None):
        """Clear the screen"""
        if color:
            self.background_color = self._normalize_color(color)
        self.ctx.clear(*self.background_color)
    
    def flush_batch(self):
        """Execute all queued draw commands in batches"""
        if not self.draw_queue:
            return
        
        batches = self.batch_manager.group_commands(self.draw_queue)
        for batch in batches:
            self.batch_manager.render_batch(batch)
        
        self.draw_queue.clear()
        self.draw_order_counter = 0
    
    # ===== 2D Drawing Methods (delegate to Primitives2D) =====
    
    def circle(self, center: Tuple[float, float], radius: float):
        """Draw a circle"""
        self.primitives_2d.circle(center, radius)
    
    def rectangle(self, pos1: Tuple[float, float], pos2: Tuple[float, float]):
        """Draw a rectangle"""
        self.primitives_2d.rectangle(pos1, pos2)
    
    def line(self, pos1: Tuple[float, float], pos2: Tuple[float, float]):
        """Draw a line"""
        self.primitives_2d.line(pos1, pos2)
    
    def polyline(self, points, closed: bool = False):
        """Draw a polyline"""
        self.primitives_2d.polyline(points, closed)

    def polygon(self, points):
        """Draw a polygon"""
        self.primitives_2d.polygon(points)
    
    # ===== 3D Drawing Methods (delegate to Primitives3D) =====
    
    def sphere(self, radius: float = 1.0, position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D sphere"""
        self.primitives_3d.sphere(radius, position)
    
    def box(self, size: Tuple[float, float, float] = (0, 0, 0),
            position: Tuple[float, float, float] = (0, 0, 0),
            texture_layers=None):
        """Draw a 3D box"""
        self.primitives_3d.box(size, position, texture_layers)
    
    def line_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]):
        """Draw a line in 3D space"""
        self.primitives_3d.line_3d(pos1, pos2)
    
    def polyline_3d(self, points, closed: bool = False):
        """Draw a 3D polyline"""
        self.primitives_3d.polyline_3d(points, closed)
        
    
    # ===== Layer Methods (delegate to LayerManager) =====
    
    def get_layer(self) -> int:
        """Create a new layer"""
        return self.layer_manager.create_layer()
    
    def begin_layer(self, layer_id: int):
        """Start rendering to a specific layer"""
        self.layer_manager.begin_layer(layer_id)
    
    def end_layer(self):
        """Stop rendering to layer"""
        self.layer_manager.end_layer()
    
    def draw_layer(self, layer_id: int, alpha: float = 1.0, x: int = 0, y: int = 0):
        """Draw a layer to the current render target"""
        self.layer_manager.draw_layer(layer_id, alpha, x, y)
    
    def clear_layer(self, layer_id: int, color: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        """Clear a layer"""
        self.layer_manager.clear_layer(layer_id, color)
    
    def release_layer(self, layer_id: int):
        """Free a layer's resources"""
        self.layer_manager.release_layer(layer_id)
    
    @property
    def layers(self):
        """Access to layers dict for compatibility"""
        return self.layer_manager.layers
    
    @property
    def active_layer(self):
        """Access to active layer"""
        return self.layer_manager.active_layer
    
    # ===== Effects Methods (delegate to EffectsManager) =====
    
    def apply_shader(self, fragment_shader_code: str, uniforms: dict = None, accumulate: bool = True):
        """Apply a custom fragment shader"""
        return self.effects.apply_shader(fragment_shader_code, uniforms, accumulate)
    
    def get_pixels(self, layer_id=None, components=3, flip=True, bgr=True) -> np.ndarray:
        """Get pixels from a framebuffer"""
        return self.effects.get_pixels(layer_id, components, flip, bgr)
    
    def paste(self, image: np.ndarray, position: Tuple[int, int],
              size: Optional[Tuple[int, int]] = None, alpha: float = 1.0):
        """Paste an image onto the canvas"""
        self.effects.paste(image, position, size, alpha)
    
    # ===== State Methods =====
    
    def fill(self, color: Tuple):
        """Set fill color"""
        self.fill_color = (*color, 255) if len(color) == 3 else color
        self.use_fill = True
    
    def no_fill(self):
        """Disable fill"""
        self.use_fill = False
    
    def stroke(self, color: Tuple):
        """Set stroke color"""
        self.stroke_color = (*color, 255) if len(color) == 3 else color
        self.use_stroke = True
    
    def no_stroke(self):
        """Disable stroke"""
        self.use_stroke = False
    
    def set_stroke_weight(self, weight: float):
        """Set stroke weight"""
        self._stroke_weight = weight
    
    # ===== Transform Methods =====
    
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
    
    # ===== Utility Methods =====
    
    def _normalize_color(self, color: Tuple) -> Tuple[float, float, float, float]:
        """Convert color from 0-255 to 0-1 range"""
        if len(color) == 3:
            return (color[0]/255, color[1]/255, color[2]/255, 1.0)
        elif len(color) == 4:
            return (color[0]/255, color[1]/255, color[2]/255, color[3]/255)
        else:
            raise ValueError(f"Color must be RGB or RGBA tuple, got: {color}")
    
    # ===== Mesh Loading =====
    
    def load_obj(self, filepath):
        """Load a Wavefront OBJ file"""
        vertices, normals, texcoords = [], [], []
        final_vertices, final_normals, final_texcoords = [], [], []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    texcoords.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    face_vertices = []
                    for i in range(1, len(parts)):
                        indices = parts[i].split('/')
                        v_idx = int(indices[0]) - 1
                        v = vertices[v_idx]
                        
                        if len(indices) > 1 and indices[1]:
                            vt_idx = int(indices[1]) - 1
                            vt = texcoords[vt_idx] if vt_idx < len(texcoords) else [0, 0]
                        else:
                            vt = [0, 0]
                        
                        if len(indices) > 2 and indices[2]:
                            vn_idx = int(indices[2]) - 1
                            vn = normals[vn_idx] if vn_idx < len(normals) else [0, 1, 0]
                        else:
                            vn = [0, 1, 0]
                        
                        face_vertices.append((v, vn, vt))
                    
                    for i in range(1, len(face_vertices) - 1):
                        for vertex_data in [face_vertices[0], face_vertices[i], face_vertices[i + 1]]:
                            v, vn, vt = vertex_data
                            final_vertices.extend(v)
                            final_normals.extend(vn)
                            final_texcoords.extend(vt)
        
        mesh_data = []
        for i in range(len(final_vertices) // 3):
            mesh_data.extend(final_vertices[i*3:i*3+3])
            mesh_data.extend(final_normals[i*3:i*3+3])
            mesh_data.extend(final_texcoords[i*2:i*2+2])
        
        return {
            'vertices': np.array(mesh_data, dtype='f4'),
            'vertex_count': len(final_vertices) // 3
        }
    
    def draw_mesh(self, mesh_data, texture_layer=None):
        """Draw a loaded mesh (immediate mode - not batched)"""
        vertices = mesh_data['vertices']
        
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.vertex_array(
            self.shader_3d,
            [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
        )
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.shader_3d['projection'].write(self.camera.get_projection_matrix())
        self.shader_3d['view'].write(self.camera.get_view_matrix())
        self.shader_3d['model'].write(self.transform.matrix)
        
        if texture_layer is not None and texture_layer in self.layers:
            texture = self.layers[texture_layer]['fbo'].color_attachments[0]
            texture.use(0)
            self.shader_3d['texture0'] = 0
            self.shader_3d['use_texture'] = True
        else:
            self.shader_3d['use_texture'] = False
            self.shader_3d['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
        
        self.shader_3d['camera_pos'] = tuple(self.camera.position)
        self.shader_3d['light_pos'].write(glm.vec3(self.light_pos))
        
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()