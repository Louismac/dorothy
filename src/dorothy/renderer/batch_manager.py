"""Batch rendering manager for Dorothy"""
import numpy as np
import glm
import moderngl
from typing import List
from .state import DrawCommand, DrawCommandType

class BatchManager:
    """Manages batching and rendering of draw commands"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.ctx = renderer.ctx
    
    def group_commands(self, commands: List[DrawCommand]) -> List[List[DrawCommand]]:
        """Group commands by state for efficient batching"""
        if not commands:
            return []
        
        sorted_cmds = sorted(commands, key=lambda c: c.draw_order)
        batches = []
        current_batch = [sorted_cmds[0]]
        prev_cmd = sorted_cmds[0]
        
        for cmd in sorted_cmds[1:]:
            if self._state_matches(prev_cmd, cmd):
                current_batch.append(cmd)
            else:
                batches.append(current_batch)
                current_batch = [cmd]
                prev_cmd = cmd
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _state_matches(self, cmd1: DrawCommand, cmd2: DrawCommand) -> bool:
        """Check if two commands can be batched together"""
        if cmd1.type != cmd2.type:
            return False
        
        # 3D instanced types - transforms can differ
        if cmd1.type in [DrawCommandType.SPHERE, DrawCommandType.BOX]:
            if cmd1.type == DrawCommandType.BOX:
                if cmd1.fill_vertices.shape != cmd2.fill_vertices.shape:
                    return False
                if getattr(cmd1, 'texture_layer', None) != getattr(cmd2, 'texture_layer', None):
                    return False
            return True
        
        # 3D lines - NOW INSTANCED, can batch together
        if cmd1.type == DrawCommandType.LINE_3D:
            # Lines can batch regardless of position/color (handled per-instance)
            # Just check stroke weight matches (thin vs thick use different rendering)
            if cmd1.stroke_weight <= 1.0 and cmd2.stroke_weight <= 1.0:
                return True
            if cmd1.stroke_weight > 1.0 and cmd2.stroke_weight > 1.0:
                # Thick lines batch together (grouped by weight later)
                return True
            return False
        
        # Old 3D line types (if still used)
        if cmd1.type in [DrawCommandType.THICK_LINE_3D, DrawCommandType.POLYLINE_3D]:
            if cmd1.use_stroke != cmd2.use_stroke:
                return False
            if cmd1.use_stroke and cmd1.stroke_color != cmd2.stroke_color:
                return False
            if abs(cmd1.stroke_weight - cmd2.stroke_weight) > 0.01:
                return False
            return self._transforms_equal(cmd1.transform, cmd2.transform)
        
        # 2D instanced types
        if cmd1.type in [DrawCommandType.CIRCLE, DrawCommandType.RECTANGLE, DrawCommandType.LINE]:
            if cmd1.use_stroke != cmd2.use_stroke:
                return False
            return True
        
        # Default: full state match required
        if cmd1.use_fill != cmd2.use_fill:
            return False
        if cmd1.use_fill and cmd1.color != cmd2.color:
            return False
        if cmd1.use_stroke != cmd2.use_stroke:
            return False
        if cmd1.use_stroke:
            if cmd1.stroke_color != cmd2.stroke_color:
                return False
            if abs(cmd1.stroke_weight - cmd2.stroke_weight) > 0.01:
                return False
        
        return self._transforms_equal(cmd1.transform, cmd2.transform)
    
    def _transforms_equal(self, t1, t2, epsilon=1e-5) -> bool:
        """Check if two transforms are equal"""
        if isinstance(t1, glm.mat4):
            a1 = np.array([t1[i][j] for i in range(4) for j in range(4)])
        else:
            a1 = np.array(t1).flatten()
        
        if isinstance(t2, glm.mat4):
            a2 = np.array([t2[i][j] for i in range(4) for j in range(4)])
        else:
            a2 = np.array(t2).flatten()
        
        return np.allclose(a1, a2, atol=epsilon)
    
    def render_batch(self, commands: List[DrawCommand]):
        """Render a batch of commands"""
        if not commands:
            return
        
        first_cmd = commands[0]
        if not first_cmd.is_3d:
            self._render_2d_batch(commands)
        else:
            self._render_3d_batch(commands)
    
    def _render_2d_batch(self, commands: List[DrawCommand]):
        """Render 2D batch - delegates to specific renderers"""
        first_cmd = commands[0]
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        if first_cmd.type == DrawCommandType.CIRCLE:
            self._render_circle_batch(commands)
        elif first_cmd.type == DrawCommandType.RECTANGLE:
            self._render_rectangle_batch(commands)
        elif first_cmd.type == DrawCommandType.LINE:
            self._render_line_batch(commands)
        else:
            self._render_legacy_2d_batch(commands)
    
    def _render_circle_batch(self, commands: List[DrawCommand]):
        """Render instanced circles"""
        first_cmd = commands[0]
        max_radius = max(cmd.radius for cmd in commands if cmd.radius)
        segments = 8 if max_radius < 2 else 16 if max_radius < 10 else 32 if max_radius < 50 else 48
        
        # Fill
        if first_cmd.use_fill:
            fill_data = []
            for cmd in commands:
                if cmd.use_fill:
                    color = self.renderer._normalize_color(cmd.color)
                    fill_data.extend([cmd.center[0], cmd.center[1], cmd.radius, *color])
            
            if fill_data:
                self._render_instanced_geometry(
                    self.renderer.geometry.get_unit_circle_vbo(segments),
                    np.array(fill_data, dtype='f4'),
                    self.renderer.shader_2d_instanced,
                    '2f', '2f 1f 4f /i',
                    ['in_position'], ['instance_center', 'instance_radius', 'instance_color'],
                    len(fill_data) // 7
                )
        
        # Stroke
        if first_cmd.use_stroke:
            strokes_by_weight = {}
            for cmd in commands:
                if cmd.use_stroke:
                    strokes_by_weight.setdefault(cmd.stroke_weight, []).append(cmd)
            
            for weight, stroke_cmds in strokes_by_weight.items():
                if weight >= 1.0:
                    avg_radius = sum(cmd.radius for cmd in stroke_cmds) / len(stroke_cmds)
                    thickness_ratio = weight / avg_radius
                    stroke_data = []
                    for cmd in stroke_cmds:
                        color = self.renderer._normalize_color(cmd.stroke_color)
                        stroke_data.extend([cmd.center[0], cmd.center[1], cmd.radius, *color])
                    
                    if stroke_data:
                        self._render_instanced_geometry(
                            self.renderer.geometry.get_unit_circle_stroke_vbo(segments, thickness_ratio),
                            np.array(stroke_data, dtype='f4'),
                            self.renderer.shader_2d_instanced,
                            '2f', '2f 1f 4f /i',
                            ['in_position'], ['instance_center', 'instance_radius', 'instance_color'],
                            len(stroke_data) // 7
                        )
    
    def _render_rectangle_batch(self, commands: List[DrawCommand]):
        """Render instanced rectangles"""
        first_cmd = commands[0]
        
        # Fill
        if first_cmd.use_fill:
            fill_data = []
            for cmd in commands:
                if cmd.use_fill:
                    x1, y1 = cmd.rect_pos1
                    x2, y2 = cmd.rect_pos2
                    color = self.renderer._normalize_color(cmd.color)
                    fill_data.extend([x1, y1, x2 - x1, y2 - y1, *color])
            
            if fill_data:
                self._render_instanced_geometry(
                    self.renderer.geometry.get_unit_rectangle_vbo(),
                    np.array(fill_data, dtype='f4'),
                    self.renderer.shader_2d_instanced_rect,
                    '2f', '2f 2f 4f /i',
                    ['in_position'], ['instance_pos', 'instance_size', 'instance_color'],
                    len(fill_data) // 8
                )
        
        # Stroke
        # In the stroke rendering section:
        if first_cmd.use_stroke:
            # Group by stroke weight
            strokes_by_weight = {}
            for cmd in commands:
                if cmd.use_stroke:
                    strokes_by_weight.setdefault(cmd.stroke_weight, []).append(cmd)
            
            for stroke_weight, stroke_cmds in strokes_by_weight.items():
                if stroke_weight >= 1.0:
                    # Calculate thickness as absolute value relative to unit rectangle
                    # For each rectangle, normalize thickness by its dimensions
                    stroke_instance_data = []
                    
                    # Create separate cache entries based on actual thickness
                    # Use a normalized thickness value
                    min_dimension = min(
                        min(abs(cmd.rect_pos2[0] - cmd.rect_pos1[0]), 
                            abs(cmd.rect_pos2[1] - cmd.rect_pos1[1])) 
                        for cmd in stroke_cmds
                    )
                    
                    # Thickness ratio relative to smallest dimension
                    thickness_ratio = stroke_weight / max(min_dimension, 1.0)
                    
                    # Clamp to prevent issues with very thin rectangles
                    thickness_ratio = min(thickness_ratio, 0.4)
                    
                    unit_stroke_vbo = self.renderer.geometry.get_unit_rectangle_stroke_vbo(thickness_ratio)
                    
                    # Build instance data
                    for cmd in stroke_cmds:
                        x1, y1 = cmd.rect_pos1
                        x2, y2 = cmd.rect_pos2
                        width = x2 - x1
                        height = y2 - y1
                        
                        color = self.renderer._normalize_color(cmd.stroke_color)
                        stroke_instance_data.extend([
                            x1, y1,           # position (2f)
                            width, height,    # size (2f)
                            *color            # color (4f)
                        ])
                    
                    if stroke_instance_data:
                        stroke_instance_array = np.array(stroke_instance_data, dtype='f4')
                        stroke_instance_buffer = self.ctx.buffer(stroke_instance_array)
                        
                        stroke_vao = self.ctx.vertex_array(
                            self.renderer.shader_2d_instanced_rect,
                            [
                                (unit_stroke_vbo, '2f', 'in_position'),
                                (stroke_instance_buffer, '2f 2f 4f /i', 'instance_pos', 'instance_size', 'instance_color'),
                            ]
                        )
                        
                        self.renderer.shader_2d_instanced_rect['projection'].write(self.renderer.camera.get_projection_matrix())
                        self.renderer.shader_2d_instanced_rect['model'].write(glm.mat4())
                        
                        stroke_vao.render(moderngl.TRIANGLES, instances=len(stroke_instance_data) // 8)
                        
                        stroke_vao.release()
                        stroke_instance_buffer.release()
                else:
                    # Thin stroke
                    pass
    
    def _render_line_batch(self, commands: List[DrawCommand]):
        """Render instanced lines"""
        first_cmd = commands[0]
        
        if first_cmd.stroke_weight <= 1.0:
            # Thin lines
            line_data = []
            for cmd in commands:
                color = self.renderer._normalize_color(cmd.stroke_color)
                line_data.extend([cmd.line_start[0], cmd.line_start[1], cmd.line_end[0], cmd.line_end[1], *color])
            
            if line_data:
                self._render_instanced_geometry(
                    self.renderer.geometry.get_unit_line_vbo(),
                    np.array(line_data, dtype='f4'),
                    self.renderer.shader_2d_instanced_line,
                    '1f', '2f 2f 4f /i',
                    ['in_position'], ['instance_start', 'instance_end', 'instance_color'],
                    len(line_data) // 8,
                    mode=moderngl.LINES
                )
        else:
            # Thick lines
            lines_by_thickness = {}
            for cmd in commands:
                lines_by_thickness.setdefault(cmd.stroke_weight, []).append(cmd)
            
            for thickness, thick_cmds in lines_by_thickness.items():
                line_data = []
                for cmd in thick_cmds:
                    color = self.renderer._normalize_color(cmd.stroke_color)
                    line_data.extend([cmd.line_start[0], cmd.line_start[1], cmd.line_end[0], cmd.line_end[1], thickness, *color])
                
                if line_data:
                    self._render_instanced_geometry(
                        self.renderer.geometry.get_unit_thick_line_vbo(),
                        np.array(line_data, dtype='f4'),
                        self.renderer.shader_2d_instanced_thick_line,
                        '2f', '2f 2f 1f 4f /i',
                        ['in_position'], ['instance_start', 'instance_end', 'instance_thickness', 'instance_color'],
                        len(line_data) // 9
                    )
    
    def _render_instanced_geometry(self, vertex_vbo, instance_data, shader, 
                                   vertex_format, instance_format,
                                   vertex_attrs, instance_attrs, 
                                   instance_count, mode=moderngl.TRIANGLES):
        """Helper to render instanced geometry"""
        instance_buffer = self.ctx.buffer(instance_data)
        
        vao = self.ctx.vertex_array(
            shader,
            [
                (vertex_vbo, vertex_format, *vertex_attrs),
                (instance_buffer, instance_format, *instance_attrs),
            ]
        )
        
        shader['projection'].write(self.renderer.camera.get_projection_matrix())
        shader['model'].write(glm.mat4())
        
        vao.render(mode, instances=instance_count)
        
        vao.release()
        instance_buffer.release()
    
    def _render_legacy_2d_batch(self, commands: List[DrawCommand]):
        """Render non-instanced 2D geometry"""
        first_cmd = commands[0]
        self.renderer.shader_2d['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_2d['model'].write(first_cmd.transform)
        
        # Fill
        if first_cmd.use_fill and first_cmd.color:
            try:
                all_verts = np.concatenate([cmd.fill_vertices for cmd in commands if len(cmd.fill_vertices) > 0])
                if len(all_verts) > 0:
                    vbo = self.ctx.buffer(all_verts)
                    vao = self.ctx.simple_vertex_array(self.renderer.shader_2d, vbo, 'in_position')
                    self.renderer.shader_2d['color'].write(glm.vec4(*self.renderer._normalize_color(first_cmd.color)))
                    vao.render(moderngl.TRIANGLES)
                    vao.release()
                    vbo.release()
            except:
                pass
        
        # Stroke
        if first_cmd.use_stroke and first_cmd.stroke_color:
            try:
                stroke_verts = [cmd.stroke_vertices for cmd in commands if cmd.stroke_vertices is not None and len(cmd.stroke_vertices) > 0]
                if stroke_verts:
                    all_verts = np.concatenate(stroke_verts)
                    vbo = self.ctx.buffer(all_verts)
                    vao = self.ctx.simple_vertex_array(self.renderer.shader_2d, vbo, 'in_position')
                    self.renderer.shader_2d['color'].write(glm.vec4(*self.renderer._normalize_color(first_cmd.stroke_color)))
                    mode = moderngl.TRIANGLES if first_cmd.stroke_as_geometry else moderngl.LINES
                    vao.render(mode)
                    vao.release()
                    vbo.release()
            except:
                pass
        
    def _render_3d_batch(self, commands: List[DrawCommand]):
        """Render 3D batch"""
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        first_cmd = commands[0]
        
        if first_cmd.type == DrawCommandType.SPHERE:
            self._render_sphere_batch(commands)
        elif first_cmd.type == DrawCommandType.BOX:
            self._render_box_batch(commands)
        elif first_cmd.type == DrawCommandType.LINE_3D:
            self._render_line_3d_batch_instanced(commands)
        elif first_cmd.type == DrawCommandType.THICK_LINE_3D:
            self._render_thick_line_3d_batch(commands)
        else:
            self._render_line_3d_batch(commands)

    def _render_line_3d_batch_instanced(self, commands: List[DrawCommand]):
        """Render instanced 3D lines"""
        first_cmd = commands[0]
        
        if first_cmd.stroke_weight <= 1.0:
            # Thin lines
            line_data = []
            for cmd in commands:
                color = self.renderer._normalize_color(cmd.stroke_color)
                line_data.extend([
                    cmd.line_start[0], cmd.line_start[1], cmd.line_start[2],  # start (3f)
                    cmd.line_end[0], cmd.line_end[1], cmd.line_end[2],        # end (3f)
                    *color                                                      # color (4f)
                ])
            
            if line_data:
                instance_array = np.array(line_data, dtype='f4')
                instance_buffer = self.ctx.buffer(instance_array)
                
                vao = self.ctx.vertex_array(
                    self.renderer.shader_3d_instanced_line,
                    [
                        (self.renderer.geometry.get_unit_line_3d_vbo(), '1f', 'in_position'),
                        (instance_buffer, '3f 3f 4f /i', 'instance_start', 'instance_end', 'instance_color'),
                    ]
                )
                
                self.renderer.shader_3d_instanced_line['view'].write(self.renderer.camera.get_view_matrix())
                self.renderer.shader_3d_instanced_line['projection'].write(self.renderer.camera.get_projection_matrix())
                
                self.ctx.line_width = first_cmd.stroke_weight
                vao.render(moderngl.LINES, instances=len(line_data) // 10)
                
                vao.release()
                instance_buffer.release()
        else:
            # Thick lines - group by thickness
            lines_by_thickness = {}
            for cmd in commands:
                lines_by_thickness.setdefault(cmd.stroke_weight, []).append(cmd)
            
            for thickness, thick_cmds in lines_by_thickness.items():
                line_data = []
                for cmd in thick_cmds:
                    color = self.renderer._normalize_color(cmd.stroke_color)
                    scaled_thickness = thickness * 0.05  # Scale appropriately
                    line_data.extend([
                        cmd.line_start[0], cmd.line_start[1], cmd.line_start[2],  # start (3f)
                        cmd.line_end[0], cmd.line_end[1], cmd.line_end[2],        # end (3f)
                        scaled_thickness,                                          # thickness (1f)
                        *color                                                      # color (4f)
                    ])
                
                if line_data:
                    instance_array = np.array(line_data, dtype='f4')
                    instance_buffer = self.ctx.buffer(instance_array)
                    
                    vao = self.ctx.vertex_array(
                        self.renderer.shader_3d_instanced_thick_line,
                        [
                            (self.renderer.geometry.get_unit_thick_line_3d_vbo(), '3f 3f', 'in_position', 'in_normal'),
                            (instance_buffer, '3f 3f 1f 4f /i', 'instance_start', 'instance_end', 'instance_thickness', 'instance_color'),
                        ]
                    )
                    
                    self.renderer.shader_3d_instanced_thick_line['view'].write(self.renderer.camera.get_view_matrix())
                    self.renderer.shader_3d_instanced_thick_line['projection'].write(self.renderer.camera.get_projection_matrix())
                    
                    vao.render(moderngl.TRIANGLES, instances=len(line_data) // 11)
                    
                    vao.release()
                    instance_buffer.release()
    
    def _render_sphere_batch(self, commands: List[DrawCommand]):
        """Render instanced spheres"""
        instance_data = []
        for cmd in commands:
            matrix_flat = np.array(cmd.transform, dtype='f4').T.flatten()
            color = self.renderer._normalize_color(cmd.color)
            instance_data.append(np.concatenate([matrix_flat, color]))
        
        instance_array = np.array(instance_data, dtype='f4')
        instance_buffer = self.ctx.buffer(instance_array)
        
        vao = self.ctx.vertex_array(
            self.renderer.shader_3d_instanced,
            [
                (self.renderer.geometry.sphere_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                (instance_buffer, '16f 4f /i', 'instance_model', 'instance_color'),
            ]
        )
        
        self.renderer.shader_3d_instanced['view'].write(self.renderer.camera.get_view_matrix())
        self.renderer.shader_3d_instanced['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_3d_instanced['light_pos'].write(glm.vec3(self.renderer.light_pos))
        self.renderer.shader_3d_instanced['camera_pos'].write(self.renderer.camera.position)
        self.renderer.shader_3d_instanced['use_lighting'] = self.renderer.use_lighting
        
        vao.render(moderngl.TRIANGLES, instances=len(commands))
        
        vao.release()
        instance_buffer.release()
    
    def _render_box_batch(self, commands: List[DrawCommand]):
        """Render instanced boxes"""
        first_cmd = commands[0]
        box_vbo = self.ctx.buffer(first_cmd.fill_vertices)
        
        instance_data = []
        for cmd in commands:
            matrix_flat = np.array(cmd.transform, dtype='f4').T.flatten()
            color = self.renderer._normalize_color(cmd.color)
            instance_data.append(np.concatenate([matrix_flat, color]))
        
        instance_array = np.array(instance_data, dtype='f4')
        instance_buffer = self.ctx.buffer(instance_array)
        
        vao = self.ctx.vertex_array(
            self.renderer.shader_3d_instanced,
            [
                (box_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                (instance_buffer, '16f 4f /i', 'instance_model', 'instance_color'),
            ]
        )
        
        self.renderer.shader_3d_instanced['view'].write(self.renderer.camera.get_view_matrix())
        self.renderer.shader_3d_instanced['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_3d_instanced['light_pos'].write(glm.vec3(self.renderer.light_pos))
        self.renderer.shader_3d_instanced['camera_pos'].write(self.renderer.camera.position)
        self.renderer.shader_3d_instanced['use_lighting'] = self.renderer.use_lighting
        
        if first_cmd.texture_layer and first_cmd.texture_layer in self.renderer.layers:
            texture = self.renderer.layers[first_cmd.texture_layer]['fbo'].color_attachments[0]
            texture.use(0)
            self.renderer.shader_3d_instanced['texture0'] = 0
            self.renderer.shader_3d_instanced['use_texture'] = True
        else:
            self.renderer.shader_3d_instanced['use_texture'] = False
        
        vao.render(moderngl.TRIANGLES, instances=len(commands))
        
        vao.release()
        instance_buffer.release()
        box_vbo.release()
    
    def _render_thick_line_3d_batch(self, commands: List[DrawCommand]):
        """Render thick 3D lines"""
        all_vertices = np.concatenate([cmd.stroke_vertices for cmd in commands])
        vbo = self.ctx.buffer(all_vertices)
        vao = self.ctx.vertex_array(
            self.renderer.shader_3d,
            [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
        )
        
        first_cmd = commands[0]
        self.renderer.shader_3d['model'].write(first_cmd.transform)
        self.renderer.shader_3d['view'].write(self.renderer.camera.get_view_matrix())
        self.renderer.shader_3d['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_3d['color'].write(glm.vec4(*self.renderer._normalize_color(first_cmd.stroke_color)))
        self.renderer.shader_3d['use_lighting'] = self.renderer.use_lighting
        self.renderer.shader_3d['use_texture'] = False
        
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()
    
    def _render_line_3d_batch(self, commands: List[DrawCommand]):
        """Render thin 3D lines"""
        all_vertices = []
        for cmd in commands:
            verts = cmd.stroke_vertices
            for i in range(0, len(verts), 3):
                all_vertices.extend([verts[i], verts[i+1], verts[i+2], 0.0, 0.0, 0.0, 0.0, 0.0])
        
        vertices_array = np.array(all_vertices, dtype='f4')
        vbo = self.ctx.buffer(vertices_array)
        vao = self.ctx.vertex_array(
            self.renderer.shader_3d,
            [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')]
        )
        
        first_cmd = commands[0]
        self.renderer.shader_3d['model'].write(first_cmd.transform)
        self.renderer.shader_3d['view'].write(self.renderer.camera.get_view_matrix())
        self.renderer.shader_3d['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_3d['color'].write(glm.vec4(*self.renderer._normalize_color(first_cmd.stroke_color)))
        self.renderer.shader_3d['use_lighting'] = False
        self.renderer.shader_3d['use_texture'] = False
        
        mode = moderngl.LINES if first_cmd.type == DrawCommandType.LINE_3D else moderngl.LINE_STRIP
        vao.render(mode)
        vao.release()
        vbo.release()