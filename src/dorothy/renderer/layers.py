"""Layer management for Dorothy renderer"""
import moderngl
import numpy as np
from typing import Tuple, Optional

class LayerManager:
    """Manages rendering layers (framebuffers)"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.ctx = renderer.ctx
        self.layers = {}
        self.layer_stack = []
        self.layer_counter = 0
        self.active_layer = None
        self.last_fbo = None
    
    def create_layer(self) -> int:
        """Create a new layer and return its ID"""
        layer_id = self.layer_counter
        self.layer_counter += 1
        
        texture = self.ctx.texture((self.renderer.width, self.renderer.height), 4)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        
        depth = self.ctx.depth_renderbuffer((self.renderer.width, self.renderer.height))
        fbo = self.ctx.framebuffer(color_attachments=[texture], depth_attachment=depth)
        
        self.layers[layer_id] = {
            'fbo': fbo,
            'texture': texture,
            'depth': depth
        }
        
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        
        print(f"Created layer {layer_id}: {self.renderer.width}x{self.renderer.height}")
        
        return layer_id
    
    def begin_layer(self, layer_id: int):
        """Start rendering to a specific layer"""
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        self.layer_stack.append(self.active_layer)
        self.active_layer = layer_id
        fbo = self.layers[layer_id]['fbo']
        self._ensure_fbo(fbo)
        fbo.use()
        self.ctx.viewport = (0, 0, self.renderer.width, self.renderer.height)
    
    def end_layer(self):
        """Stop rendering to layer, return to previous layer"""
        if self.active_layer is None:
            return
        
        if self.renderer.enable_batching and len(self.renderer.draw_queue) > 0:
            self._flush_for_fbo_change()
        
        if self.layer_stack:
            prev_layer = self.layer_stack.pop()
            
            if prev_layer is not None:
                self.active_layer = prev_layer
                self.layers[prev_layer]['fbo'].use()
            else:
                self.active_layer = None
                self.last_fbo = None
                self.ctx.screen.use()
        else:
            self.active_layer = None
            self.last_fbo = None
            self.ctx.screen.use()
        
        self.ctx.viewport = (0, 0, self.renderer.width, self.renderer.height)
    
    def draw_layer(self, layer_id: int, alpha: float = 1.0, x: int = 0, y: int = 0):
        """Draw a layer to the current render target"""
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        texture = self.layers[layer_id]['texture']
        texture.use(0)
        
        has_transform = not np.allclose(self.renderer.transform.matrix, self.renderer._identity_matrix)
        
        if has_transform:
            shader = self.renderer.shader_texture_transform
            vertices = np.array([
                0, 0, 0.0, 1.0,
                self.renderer.width, 0, 1.0, 1.0,
                self.renderer.width, self.renderer.height, 1.0, 0.0,
                0, 0, 0.0, 1.0,
                self.renderer.width, self.renderer.height, 1.0, 0.0,
                0, self.renderer.height, 0.0, 0.0,
            ], dtype='f4')
            
            vbo = self.ctx.buffer(vertices)
            vao = self.ctx.simple_vertex_array(shader, vbo, 'in_position', 'in_texcoord_0')
            
            shader['projection'].write(self.renderer.camera.get_projection_matrix())
            shader['model'].write(self.renderer.transform.matrix)
            shader['texture0'] = 0
            shader['alpha'] = alpha
            
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            shader = self.renderer.shader_texture
            shader['texture0'] = 0
            shader['alpha'] = alpha
            self.renderer.quad_vao.render(moderngl.TRIANGLES)
    
    def clear_layer(self, layer_id: int, color: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        """Clear a layer with a specific color"""
        if self.renderer.enable_batching and len(self.renderer.draw_queue) > 0:
            self._flush_for_fbo_change()
        
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        fbo = self.layers[layer_id]['fbo']
        fbo.use()
        fbo.clear(*color)
    
    def release_layer(self, layer_id: int):
        """Free a layer's resources"""
        if layer_id in self.layers:
            self.layers[layer_id]['fbo'].release()
            self.layers[layer_id]['texture'].release()
            del self.layers[layer_id]
    
    def _ensure_fbo(self, target_fbo):
        """Ensure correct FBO is bound, flushing if needed"""
        if target_fbo != self.last_fbo:
            if self.renderer.enable_batching and len(self.renderer.draw_queue) > 0:
                self._flush_for_fbo_change()
            
            if target_fbo is None:
                self.ctx.screen.use()
            else:
                target_fbo.use()
            
            self.last_fbo = target_fbo
    
    def _flush_for_fbo_change(self):
        """Flush batch because FBO is changing"""
        batches = self.renderer.batch_manager.group_commands(self.renderer.draw_queue)
        for batch in batches:
            self.renderer.batch_manager.render_batch(batch)
        
        self.renderer.draw_queue.clear()
        self.renderer.draw_order_counter = 0