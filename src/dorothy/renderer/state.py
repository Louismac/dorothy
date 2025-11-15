"""State management classes for Dorothy renderer"""
import glm
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

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
    texture_layer: dict = None
    is_3d: bool = False 
    # Instanced rendering parameters
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    rect_pos1: Optional[Tuple[float, float]] = None
    rect_pos2: Optional[Tuple[float, float]] = None
    line_start: Optional[Tuple[float, float]] = None
    line_end: Optional[Tuple[float, float]] = None


class Transform:
    """Manages transformation matrices""" 
    def __init__(self):
        self.matrix = glm.mat4(1.0)
        self.stack = []
    
    def push(self):
        self.stack.append(glm.mat4(self.matrix))
    
    def pop(self):
        if self.stack:
            self.matrix = self.stack.pop()
    
    def reset(self):
        self.matrix = glm.mat4(1.0)
    
    def translate(self, x: float, y: float, z: float = 0):
        self.matrix = glm.translate(self.matrix, glm.vec3(x, y, z))
    
    def rotate(self, angle: float, x: float = 0, y: float = 0, z: float = 1):
        self.matrix = glm.rotate(self.matrix, angle, glm.vec3(x, y, z))
    
    def scale(self, x: float, y: float = None, z: float = None):
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
        self.mode = '2d'
        self.width = width
        self.height = height
    
    def get_view_matrix(self):
        return glm.lookAt(self.position, self.target, self.up)
    
    def get_projection_matrix(self):
        if self.mode == '3d':
            return glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
        else:
            return glm.ortho(0, self.width, self.height, 0, -1, 1)