"""3D primitive drawing methods for Dorothy"""
import numpy as np
from typing import Tuple
from .state import DrawCommand, DrawCommandType
import moderngl
import glm

class Primitives3D:
    """3D drawing primitives"""
    
    def __init__(self, renderer):
        self.renderer = renderer
    
    def sphere(self, radius: float = 1.0, position: Tuple[float, float, float] = (0, 0, 0)):
        """Draw a 3D sphere"""
        self.renderer.transform.push()
        self.renderer.transform.translate(position[0], position[1], position[2])
        self.renderer.transform.scale(radius)
        
        self._draw_3d_geometry(None, DrawCommandType.SPHERE)
        
        self.renderer.transform.pop()
    
    def box(self, size: Tuple[float, float, float] = (0, 0, 0), 
            position: Tuple[float, float, float] = (0, 0, 0),
            texture_layers=None):
        """Draw a 3D box"""
        w, h, d = size
        
        # Normalize texture layers
        if isinstance(texture_layers, int):
            uniform_texture = texture_layers
        elif isinstance(texture_layers, dict):
            unique_textures = set(texture_layers.values())
            if len(unique_textures) == 1:
                uniform_texture = list(unique_textures)[0]
            else:
                self._draw_box_immediate(w, h, d, position, texture_layers)
                return
        else:
            uniform_texture = None
        
        faces_data = self._create_box_faces(w, h, d)
        all_vertices = np.concatenate(list(faces_data.values()))
        
        self.renderer.transform.push()
        self.renderer.transform.translate(position[0], position[1], position[2])
        
        self._draw_3d_geometry(all_vertices, DrawCommandType.BOX, texture_layer=uniform_texture)
        
        self.renderer.transform.pop()
    
    def line_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]):
        """Draw a line in 3D space"""
        if self.renderer._stroke_weight == 1:
            vertices = np.array([pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]], dtype='f4')
            self._draw_3d_geometry(vertices, DrawCommandType.LINE_3D)
        else:
            thickness = self.renderer._stroke_weight * 0.01
            vertices = self._create_3d_line_geometry(pos1, pos2, thickness)
            if len(vertices) > 0:
                self._draw_3d_geometry(vertices, DrawCommandType.THICK_LINE_3D)
    
    def polyline_3d(self, points, closed: bool = False):
        """Draw a 3D polyline"""
        if len(points) < 2:
            return
        
        if self.renderer._stroke_weight == 1:
            vertices = []
            for x, y, z in points:
                vertices.extend([x, y, z])
            if closed:
                vertices.extend([points[0][0], points[0][1], points[0][2]])
            vertices = np.array(vertices, dtype='f4')
            self._draw_3d_geometry(vertices, DrawCommandType.POLYLINE_3D)
        else:
            thickness = self.renderer._stroke_weight * 0.01
            vertices = self._create_3d_polyline_geometry(points, thickness, closed)
            if len(vertices) > 0:
                self._draw_3d_geometry(vertices, DrawCommandType.THICK_LINE_3D)
    
    def _draw_3d_geometry(self, geom, type, texture_layer=None):
        """Queue 3D geometry for rendering"""
        filled = type in [DrawCommandType.SPHERE, DrawCommandType.BOX]
        
        cmd = DrawCommand(
            type=type,
            fill_vertices=geom if filled else None,
            stroke_vertices=None if filled else geom,
            color=self.renderer.fill_color if self.renderer.use_fill else None,
            use_fill=self.renderer.use_fill,
            use_stroke=self.renderer.use_stroke,
            stroke_weight=self.renderer._stroke_weight,
            stroke_color=self.renderer.stroke_color if self.renderer.use_stroke else None,
            transform=self.renderer.transform.matrix,
            layer_id=self.renderer.active_layer,
            draw_order=self.renderer.draw_order_counter,
            is_3d=True,
            texture_layer=texture_layer
        )
        
        self.renderer.draw_order_counter += 1
        self.renderer.draw_queue.append(cmd)
    
    def _create_box_faces(self, w, h, d):
        """Create box face geometry"""
        return {
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
    
    def _create_3d_line_geometry(self, pos1, pos2, thickness=0.05):
        """Create rectangular tube between two points"""
        p1 = np.array(pos1, dtype='f4')
        p2 = np.array(pos2, dtype='f4')
        
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 0.0001:
            return np.array([], dtype='f4')
        
        direction = direction / length
        
        if abs(direction[1]) > 0.99:
            up = np.array([1.0, 0.0, 0.0], dtype='f4')
        else:
            up = np.array([0.0, 1.0, 0.0], dtype='f4')
        
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right) * (thickness / 2)
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up) * (thickness / 2)
        
        corners = [
            p1 - right - up, p1 + right - up, p1 + right + up, p1 - right + up,
            p2 - right - up, p2 + right - up, p2 + right + up, p2 - right + up,
        ]
        
        vertices = []
        
        def add_quad(i1, i2, i3, i4, normal):
            for idx in [i1, i2, i3, i1, i3, i4]:
                vertices.extend(corners[idx])
                vertices.extend(normal)
                vertices.extend([0.0, 0.0])
        
        add_quad(0, 1, 2, 3, -direction)
        add_quad(5, 4, 7, 6, direction)
        add_quad(0, 4, 5, 1, -up)
        add_quad(2, 6, 7, 3, up)
        add_quad(1, 5, 6, 2, right)
        add_quad(4, 0, 3, 7, -right)
        
        return np.array(vertices, dtype='f4')
    
    def _create_3d_polyline_geometry(self, points, thickness=0.05, closed=False):
        """Create geometry for a polyline"""
        if len(points) < 2:
            return np.array([], dtype='f4')
        
        all_vertices = []
        
        for i in range(len(points) - 1):
            segment_verts = self._create_3d_line_geometry(points[i], points[i+1], thickness)
            if len(segment_verts) > 0:
                all_vertices.append(segment_verts)
        
        if closed and len(points) > 2:
            segment_verts = self._create_3d_line_geometry(points[-1], points[0], thickness)
            if len(segment_verts) > 0:
                all_vertices.append(segment_verts)
        
        if len(all_vertices) == 0:
            return np.array([], dtype='f4')
        
        return np.concatenate(all_vertices)
    
    def _draw_box_immediate(self, w, h, d, position, tex_dict):
        """Render a box with per-face textures immediately (can't batch)"""
        faces_data = self._create_box_faces(w, h, d)
        
        self.renderer.ctx.enable(moderngl.BLEND)
        self.renderer.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.renderer.ctx.enable(moderngl.DEPTH_TEST)
        
        self.renderer.transform.push()
        self.renderer.transform.translate(position[0], position[1], position[2])
        
        for face_name, vertices in faces_data.items():
            layer_id = tex_dict[face_name]
            
            vbo = self.renderer.ctx.buffer(vertices)
            shader = self.renderer.shader_3d
            vao = self.renderer.ctx.vertex_array(
                shader,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
            )
            
            # Set uniforms
            shader['projection'].write(self.renderer.camera.get_projection_matrix())
            shader['view'].write(self.renderer.camera.get_view_matrix())
            shader['model'].write(self.renderer.transform.matrix)
            
            # Texture or solid color
            if layer_id is not None and layer_id in self.renderer.layers:
                texture = self.renderer.layers[layer_id]['fbo'].color_attachments[0]
                texture.use(0)
                shader['texture0'] = 0
                shader['use_texture'] = True
            else:
                shader['use_texture'] = False
                shader['color'].write(glm.vec4(*self._normalize_color(self.fill_color)))
            
            shader['use_lighting'] = self.renderer.use_lighting
            shader['light_pos'].write(glm.vec3(self.renderer.light_pos))
            shader['camera_pos'].write(self.renderer.camera.position)
            
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()
        
        self.renderer.transform.pop()