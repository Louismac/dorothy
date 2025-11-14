"""2D primitive drawing methods for Dorothy"""
import numpy as np
import glm
from typing import Tuple
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