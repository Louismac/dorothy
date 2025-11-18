"""Cached geometry generation for Dorothy renderer"""
import numpy as np

class GeometryCache:
    """Manages cached geometry for efficient rendering"""
    
    def __init__(self, ctx):
        self.ctx = ctx
        self._circle_cache = {}
        self._circle_vbo_cache = {}
        self._circle_stroke_cache = {}
        self._circle_stroke_vbo_cache = {}
        self._unit_rect_vbo = None
        self._unit_rect_stroke_cache = {}
        self._unit_line_vbo = None
        self._unit_thick_line_vbo = None
        self.sphere_vbo = None
        self.sphere_ibo = None

    def get_unit_line_3d_vbo(self):
        """Get cached unit 3D line VBO (two points: 0 and 1)"""
        if not hasattr(self, '_unit_line_3d_vbo'):
            self._unit_line_3d_vbo = self.ctx.buffer(np.array([0.0, 1.0], dtype='f4'))
        return self._unit_line_3d_vbo
    
    def get_unit_thick_line_3d_vbo(self):
        """Get cached unit thick 3D line geometry (rectangular tube)"""
        if not hasattr(self, '_unit_thick_line_3d_vbo'):
            # Rectangular tube from (0,0,0) to (1,0,0)
            # We need vertices with position AND offset information
            # Format: x (0 to 1), y_offset (-0.5 to 0.5), z_offset (-0.5 to 0.5)
            
            vertices = []
            
            # Each vertex is: x_pos, y_offset, z_offset, normal_x, normal_y, normal_z
            # Create a rectangular tube with 4 sides
            
            # Front face (z = -0.5)
            vertices.extend([
                0.0, -0.5, -0.5,  0, 0, -1,
                1.0, -0.5, -0.5,  0, 0, -1,
                1.0,  0.5, -0.5,  0, 0, -1,
                0.0, -0.5, -0.5,  0, 0, -1,
                1.0,  0.5, -0.5,  0, 0, -1,
                0.0,  0.5, -0.5,  0, 0, -1,
            ])
            
            # Back face (z = 0.5)
            vertices.extend([
                1.0, -0.5,  0.5,  0, 0, 1,
                0.0, -0.5,  0.5,  0, 0, 1,
                0.0,  0.5,  0.5,  0, 0, 1,
                1.0, -0.5,  0.5,  0, 0, 1,
                0.0,  0.5,  0.5,  0, 0, 1,
                1.0,  0.5,  0.5,  0, 0, 1,
            ])
            
            # Bottom face (y = -0.5)
            vertices.extend([
                0.0, -0.5, -0.5,  0, -1, 0,
                0.0, -0.5,  0.5,  0, -1, 0,
                1.0, -0.5,  0.5,  0, -1, 0,
                0.0, -0.5, -0.5,  0, -1, 0,
                1.0, -0.5,  0.5,  0, -1, 0,
                1.0, -0.5, -0.5,  0, -1, 0,
            ])
            
            # Top face (y = 0.5)
            vertices.extend([
                0.0,  0.5,  0.5,  0, 1, 0,
                0.0,  0.5, -0.5,  0, 1, 0,
                1.0,  0.5, -0.5,  0, 1, 0,
                0.0,  0.5,  0.5,  0, 1, 0,
                1.0,  0.5, -0.5,  0, 1, 0,
                1.0,  0.5,  0.5,  0, 1, 0,
            ])
            
            self._unit_thick_line_3d_vbo = self.ctx.buffer(np.array(vertices, dtype='f4'))
        return self._unit_thick_line_3d_vbo
    
    def get_unit_circle(self, segments):
        """Get cached unit circle (radius=1, center=0,0)"""
        if segments not in self._circle_cache:
            fill_verts = []
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * (i + 1) / segments
                x1, y1 = np.cos(angle1), np.sin(angle1)
                x2, y2 = np.cos(angle2), np.sin(angle2)
                fill_verts.extend([0, 0, x1, y1, x2, y2])
            self._circle_cache[segments] = np.array(fill_verts, dtype='f4')
        return self._circle_cache[segments]
    
    def get_unit_circle_vbo(self, segments):
        """Get cached unit circle VBO"""
        if segments not in self._circle_vbo_cache:
            unit_circle = self.get_unit_circle(segments)
            self._circle_vbo_cache[segments] = self.ctx.buffer(unit_circle)
        return self._circle_vbo_cache[segments]
    
    def get_unit_circle_stroke(self, segments, thickness_ratio):
        """Get cached unit circle stroke geometry"""
        cache_key = (segments, round(thickness_ratio, 3))
        if cache_key not in self._circle_stroke_cache:
            half_thick = thickness_ratio / 2
            outer_radius = 1.0 + half_thick
            inner_radius = 1.0 - half_thick
            vertices = []
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * (i + 1) / segments
                outer_x1, outer_y1 = outer_radius * np.cos(angle1), outer_radius * np.sin(angle1)
                outer_x2, outer_y2 = outer_radius * np.cos(angle2), outer_radius * np.sin(angle2)
                inner_x1, inner_y1 = inner_radius * np.cos(angle1), inner_radius * np.sin(angle1)
                inner_x2, inner_y2 = inner_radius * np.cos(angle2), inner_radius * np.sin(angle2)
                vertices.extend([outer_x1, outer_y1, outer_x2, outer_y2, inner_x2, inner_y2])
                vertices.extend([outer_x1, outer_y1, inner_x2, inner_y2, inner_x1, inner_y1])
            self._circle_stroke_cache[cache_key] = np.array(vertices, dtype='f4')
        return self._circle_stroke_cache[cache_key]
    
    def get_unit_circle_stroke_vbo(self, segments, thickness_ratio):
        """Get cached unit circle stroke VBO"""
        cache_key = (segments, round(thickness_ratio, 3))
        if cache_key not in self._circle_stroke_vbo_cache:
            stroke = self.get_unit_circle_stroke(segments, thickness_ratio)
            self._circle_stroke_vbo_cache[cache_key] = self.ctx.buffer(stroke)
        return self._circle_stroke_vbo_cache[cache_key]
    
    def get_unit_rectangle_vbo(self):
        """Get cached unit rectangle VBO (0,0 to 1,1)"""
        if self._unit_rect_vbo is None:
            vertices = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype='f4')
            self._unit_rect_vbo = self.ctx.buffer(vertices)
        return self._unit_rect_vbo
    
    def get_unit_rectangle_stroke_vbo(self, thickness_ratio):
        """Get cached unit rectangle stroke VBO
        
        Args:
            thickness_ratio: absolute thickness (not ratio)
        """
        cache_key = round(thickness_ratio, 3)
        
        if not hasattr(self, '_unit_rect_stroke_cache'):
            self._unit_rect_stroke_cache = {}
        
        if cache_key not in self._unit_rect_stroke_cache:
            # Use absolute thickness, not ratio
            half_thick = thickness_ratio / 2
            
            # Outer rectangle (expanded outward from 0,0 to 1,1)
            outer_x1, outer_y1 = -half_thick, -half_thick
            outer_x2, outer_y2 = 1.0 + half_thick, 1.0 + half_thick
            
            # Inner rectangle (shrunk inward)
            inner_x1, inner_y1 = half_thick, half_thick
            inner_x2, inner_y2 = 1.0 - half_thick, 1.0 - half_thick
            
            vertices = []
            
            # Top edge quad
            vertices.extend([
                outer_x1, outer_y1, outer_x2, outer_y1, inner_x2, inner_y1,
                outer_x1, outer_y1, inner_x2, inner_y1, inner_x1, inner_y1,
            ])
            
            # Right edge quad
            vertices.extend([
                outer_x2, outer_y1, outer_x2, outer_y2, inner_x2, inner_y2,
                outer_x2, outer_y1, inner_x2, inner_y2, inner_x2, inner_y1,
            ])
            
            # Bottom edge quad
            vertices.extend([
                outer_x2, outer_y2, outer_x1, outer_y2, inner_x1, inner_y2,
                outer_x2, outer_y2, inner_x1, inner_y2, inner_x2, inner_y2,
            ])
            
            # Left edge quad
            vertices.extend([
                outer_x1, outer_y2, outer_x1, outer_y1, inner_x1, inner_y1,
                outer_x1, outer_y2, inner_x1, inner_y1, inner_x1, inner_y2,
            ])
            
            self._unit_rect_stroke_cache[cache_key] = self.ctx.buffer(np.array(vertices, dtype='f4'))
        
        return self._unit_rect_stroke_cache[cache_key]
    
    def get_unit_line_vbo(self):
        """Get cached unit line VBO"""
        if self._unit_line_vbo is None:
            self._unit_line_vbo = self.ctx.buffer(np.array([0.0, 1.0], dtype='f4'))
        return self._unit_line_vbo
    
    def get_unit_thick_line_vbo(self):
        """Get cached unit thick line VBO"""
        if self._unit_thick_line_vbo is None:
            vertices = np.array([0.0, -0.5, 1.0, -0.5, 1.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0, 0.5], dtype='f4')
            self._unit_thick_line_vbo = self.ctx.buffer(vertices)
        return self._unit_thick_line_vbo
    
    def generate_sphere(self, radius=1.0, sectors=32, rings=32):
        """Generate sphere geometry"""
        vertices, normals, texcoords = [], [], []
        for ring in range(rings + 1):
            theta = ring * np.pi / rings
            sin_theta, cos_theta = np.sin(theta), np.cos(theta)
            for sector in range(sectors + 1):
                phi = sector * 2 * np.pi / sectors
                sin_phi, cos_phi = np.sin(phi), np.cos(phi)
                x, y, z = radius * sin_theta * cos_phi, radius * cos_theta, radius * sin_theta * sin_phi
                vertices.extend([x, y, z])
                normals.extend([sin_theta * cos_phi, cos_theta, sin_theta * sin_phi])
                texcoords.extend([sector / sectors, ring / rings])
        
        triangle_verts, triangle_normals, triangle_texcoords = [], [], []
        for ring in range(rings):
            for sector in range(sectors):
                current = ring * (sectors + 1) + sector
                next_ring = current + sectors + 1
                for idx in [current, next_ring, current + 1, current + 1, next_ring, next_ring + 1]:
                    triangle_verts.extend(vertices[idx*3:idx*3+3])
                    triangle_normals.extend(normals[idx*3:idx*3+3])
                    triangle_texcoords.extend(texcoords[idx*2:idx*2+2])
        
        return (np.array(triangle_verts, dtype='f4'),
                np.array(triangle_normals, dtype='f4'),
                np.array(triangle_texcoords, dtype='f4'))
    
    def initialize_sphere_vbo(self):
        """Initialize sphere VBO"""
        verts, normals, texcoords = self.generate_sphere()
        vertex_data = []
        num_verts = len(verts) // 3
        for i in range(num_verts):
            vertex_data.extend(verts[i*3:i*3+3])
            vertex_data.extend(normals[i*3:i*3+3])
            vertex_data.extend(texcoords[i*2:i*2+2])
        self.sphere_vbo = self.ctx.buffer(np.array(vertex_data, dtype='f4'))
        self.sphere_ibo = None