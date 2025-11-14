"""2D primitive drawing methods for Dorothy"""
import numpy as np
import glm
from typing import Tuple,List
import moderngl
from .state import DrawCommand, DrawCommandType

class Primitives2D:
    """2D drawing primitives"""
    
    def __init__(self, renderer):
        self.renderer = renderer
    
    def circle(self, center: Tuple[float, float], radius: float):
        """Draw a circle"""
        if self.renderer.enable_batching:
            transformed_center = self.renderer.transform.matrix * glm.vec4(center[0], center[1], 0.0, 1.0)
            world_center = (transformed_center.x, transformed_center.y)
            scale = glm.length(glm.vec3(self.renderer.transform.matrix[0]))
            world_radius = radius * scale
            
            cmd = DrawCommand(
                type=DrawCommandType.CIRCLE,
                center=world_center,
                radius=world_radius,
                color=self.renderer.fill_color if self.renderer.use_fill else None,
                use_fill=self.renderer.use_fill,
                use_stroke=self.renderer.use_stroke,
                stroke_weight=self.renderer._stroke_weight,
                stroke_color=self.renderer.stroke_color if self.renderer.use_stroke else None,
                transform=glm.mat4(),
                layer_id=self.renderer.active_layer,
                draw_order=self.renderer.draw_order_counter,
            )
            self.renderer.draw_order_counter += 1
            self.renderer.draw_queue.append(cmd)
    
    def rectangle(self, pos1: Tuple[float, float], pos2: Tuple[float, float]):
        """Draw a rectangle"""
        if self.renderer.enable_batching:
            x1, y1 = pos1
            x2, y2 = pos2
            
            p1 = self.renderer.transform.matrix * glm.vec4(x1, y1, 0.0, 1.0)
            p2 = self.renderer.transform.matrix * glm.vec4(x2, y2, 0.0, 1.0)
            
            cmd = DrawCommand(
                type=DrawCommandType.RECTANGLE,
                rect_pos1=(p1.x, p1.y),
                rect_pos2=(p2.x, p2.y),
                color=self.renderer.fill_color if self.renderer.use_fill else None,
                use_fill=self.renderer.use_fill,
                use_stroke=self.renderer.use_stroke,
                stroke_weight=self.renderer._stroke_weight,
                stroke_color=self.renderer.stroke_color if self.renderer.use_stroke else None,
                transform=glm.mat4(),
                layer_id=self.renderer.active_layer,
                draw_order=self.renderer.draw_order_counter,
            )
            
            self.renderer.draw_order_counter += 1
            self.renderer.draw_queue.append(cmd)
    
    def line(self, pos1: Tuple[float, float], pos2: Tuple[float, float]):
        """Draw a line"""
        if self.renderer.enable_batching:
            x1, y1 = pos1
            x2, y2 = pos2
            
            p1 = self.renderer.transform.matrix * glm.vec4(x1, y1, 0.0, 1.0)
            p2 = self.renderer.transform.matrix * glm.vec4(x2, y2, 0.0, 1.0)
            
            cmd = DrawCommand(
                type=DrawCommandType.LINE,
                line_start=(p1.x, p1.y),
                line_end=(p2.x, p2.y),
                use_stroke=True,
                stroke_weight=self.renderer._stroke_weight,
                stroke_color=self.renderer.stroke_color,
                transform=glm.mat4(),
                layer_id=self.renderer.active_layer,
                draw_order=self.renderer.draw_order_counter,
            )
            
            self.renderer.draw_order_counter += 1
            self.renderer.draw_queue.append(cmd)

    def polyline(self, points: List[Tuple[float, float]], closed: bool = False):
        """Draw a polyline (connected line segments)"""
        if len(points) < 2:
            return
        
        vertices = []
        for x, y in points:
            vertices.extend([x, y])
        
        if closed:
            vertices.extend([points[0][0], points[0][1]])
        
        vertices = np.array(vertices, dtype='f4')
        
        # Enable blending
        self.renderer.ctx.enable(moderngl.BLEND)
        self.renderer.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Set uniforms
        self.renderer.shader_2d['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_2d['model'].write(self.renderer.transform.matrix)
        
        if self.renderer.use_stroke:
            self.renderer.shader_2d['color'].write(glm.vec4(*self.renderer._normalize_color(self.renderer.stroke_color)))
            
            # Try thick stroke
            thick_verts = self._draw_thick_stroke(vertices, closed=closed)
            
            if thick_verts is not None:
                thick_vbo = self.renderer.ctx.buffer(thick_verts)
                thick_vao = self.renderer.ctx.simple_vertex_array(self.renderer.shader_2d, thick_vbo, 'in_position')
                thick_vao.render(moderngl.TRIANGLE_STRIP)
                thick_vao.release()
                thick_vbo.release()
            else:
                vbo = self.renderer.ctx.buffer(vertices)
                vao = self.renderer.ctx.simple_vertex_array(self.renderer.shader_2d, vbo, 'in_position')
                self.renderer.ctx.line_width = 1.0
                vao.render(moderngl.LINE_STRIP)
                vao.release()
                vbo.release()
    
    def polygon(self, points: List[Tuple[float, float]]):
        """Draw a filled polygon with proper triangulation"""
        if len(points) < 3:
            return
        
        # Triangulate the polygon
        triangulated = self._triangulate(points)
        
        if not triangulated:
            return
        
        # Create vertices array
        vertices = []
        for x, y in triangulated:
            vertices.extend([x, y])
        
        vertices = np.array(vertices, dtype='f4')
        
        vbo = self.renderer.ctx.buffer(vertices)
        vao = self.renderer.ctx.simple_vertex_array(self.renderer.shader_2d, vbo, 'in_position')
        
        # Enable blending
        self.renderer.ctx.enable(moderngl.BLEND)
        self.renderer.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Set uniforms
        self.renderer.shader_2d['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_2d['model'].write(self.renderer.transform.matrix)
        
        # Draw fill
        if self.renderer.use_fill:
            self.renderer.shader_2d['color'].write(glm.vec4(*self.renderer._normalize_color(self.renderer.fill_color)))
            vao.render(moderngl.TRIANGLES)
        
        # Draw stroke
        if self.renderer.use_stroke:
            stroke_vertices = []
            for x, y in points:
                stroke_vertices.extend([x, y])
            stroke_vertices.extend([points[0][0], points[0][1]])
            
            stroke_vbo = self.renderer.ctx.buffer(np.array(stroke_vertices, dtype='f4'))
            stroke_vao = self.renderer.ctx.simple_vertex_array(self.renderer.shader_2d, stroke_vbo, 'in_position')
            
            self.renderer.ctx.line_width = self.renderer._stroke_weight
            self.renderer.shader_2d['color'].write(glm.vec4(*self.renderer._normalize_color(self.renderer.stroke_color)))
            stroke_vao.render(moderngl.LINE_STRIP)
            
            stroke_vao.release()
            stroke_vbo.release()
        
        vao.release()
        vbo.release()
    
    def _draw_thick_stroke(self, vertices, closed=False):
        """Draw thick lines as quads instead of using line_width"""
        if self.renderer._stroke_weight <= 1.0:
            return None
        
        thickness = self.renderer._stroke_weight
        points = vertices.reshape(-1, 2)
        
        if len(points) < 2:
            return None
        
        if closed:
            points = np.vstack([points, points[0:1]])
        
        quad_vertices = []
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < 0.001:
                continue
            
            dx /= length
            dy /= length
            px = -dy * thickness / 2
            py = dx * thickness / 2
            
            quad_vertices.extend([
                x1 + px, y1 + py,
                x1 - px, y1 - py,
                x2 + px, y2 + py,
                x2 - px, y2 - py,
            ])
        
        if not quad_vertices:
            return None
        
        return np.array(quad_vertices, dtype='f4')
    
    def _triangulate(self, vertices):
        """Simple ear clipping triangulation for polygons"""
        if len(vertices) < 3:
            return []
        
        verts = list(vertices)
        triangles = []
        
        while len(verts) > 3:
            ear_found = False
            for i in range(len(verts)):
                prev = verts[i - 1]
                curr = verts[i]
                next_v = verts[(i + 1) % len(verts)]
                
                if self._is_ear(prev, curr, next_v, verts):
                    triangles.extend([prev, curr, next_v])
                    verts.pop(i)
                    ear_found = True
                    break
            
            if not ear_found:
                if len(verts) >= 3:
                    triangles.extend(verts[:3])
                break
        
        if len(verts) == 3:
            triangles.extend(verts)
        
        return triangles
    
    def _is_ear(self, prev, curr, next_v, all_verts):
        """Check if triangle is an ear"""
        cross = (curr[0] - prev[0]) * (next_v[1] - prev[1]) - (curr[1] - prev[1]) * (next_v[0] - prev[0])
        if cross <= 0:
            return False
        
        for v in all_verts:
            if v == prev or v == curr or v == next_v:
                continue
            if self._point_in_triangle(v, prev, curr, next_v):
                return False
        
        return True
    
    def _point_in_triangle(self, p, a, b, c):
        """Check if point p is inside triangle abc"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)