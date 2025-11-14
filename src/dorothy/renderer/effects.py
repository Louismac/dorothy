"""Shader effects and image operations for Dorothy"""
import numpy as np
import cv2
import moderngl
from typing import Optional, Tuple, Dict

class EffectsManager:
    """Manages shader effects and image operations"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.ctx = renderer.ctx
        self.effect_shaders = {}
        self.effect_vaos = {}
    
    def get_effect_shader(self, shader_code):
        """Get or create a cached shader program"""
        shader_hash = hash(shader_code)
        
        if shader_hash not in self.effect_shaders:
            print(f"Compiling shader...")
            
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
                self.effect_shaders[shader_hash] = program
            except Exception as e:
                print(f"Shader compilation failed: {e}")
                return None
        
        return self.effect_shaders[shader_hash]
    
    def apply_shader(self, fragment_shader_code: str, uniforms: Dict = None, accumulate: bool = True):
        """Apply a custom fragment shader to the current canvas"""
        if self.renderer.enable_batching and len(self.renderer.draw_queue) > 0:
            self._flush_for_fbo_change()
        
        custom_shader = self.get_effect_shader(fragment_shader_code)
        if custom_shader is None:
            return None
        
        layer = self.renderer.layer_manager.layers[self.renderer.layer_manager.active_layer]
        old_texture = layer['texture']
        old_fbo = layer['fbo']
        
        new_texture = self.ctx.texture((self.renderer.width, self.renderer.height), 4)
        new_fbo = self.ctx.framebuffer(color_attachments=[new_texture])
        
        shader_hash = hash(fragment_shader_code)
        if shader_hash not in self.effect_vaos:
            self.effect_vaos[shader_hash] = self.ctx.simple_vertex_array(
                custom_shader,
                self.renderer.quad_vbo,
                'in_position',
                'in_texcoord_0'
            )
        
        custom_vao = self.effect_vaos[shader_hash]
        
        new_fbo.use()
        old_texture.use(0)
        
        try:
            custom_shader['texture0'] = 0
        except KeyError:
            pass
        
        try:
            custom_shader['resolution'] = (float(self.renderer.width), float(self.renderer.height))
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
            layer['texture'] = new_texture
            layer['fbo'] = new_fbo
            old_fbo.release()
            old_texture.release()
            new_fbo.use()
            return None
        else:
            old_fbo.use()
            temp_layer_id = -1
            self.renderer.layer_manager.layers[temp_layer_id] = {
                'fbo': new_fbo,
                'texture': new_texture,
                'depth': None
            }
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            return temp_layer_id
    
    def get_pixels(self, layer_id=None, components=3, flip=True, bgr=True) -> np.ndarray:
        """Get pixels from a framebuffer as numpy array"""
        if layer_id is not None:
            if layer_id not in self.renderer.layer_manager.layers:
                raise ValueError(f"Layer {layer_id} does not exist")
            fbo = self.renderer.layer_manager.layers[layer_id]['fbo']
            pixels = fbo.read(components=components)
            w, h = fbo.size
        else:
            pixels = self.ctx.screen.read(components=components)
            w, h = self.ctx.screen.size
        
        img = np.frombuffer(pixels, dtype=np.uint8)
        img = img.reshape((h, w, components))
        
        if flip:
            img = np.flipud(img)
        
        if bgr and components == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif bgr and components == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        
        return img
    
    def paste(self, image: np.ndarray, position: Tuple[int, int], 
              size: Optional[Tuple[int, int]] = None, alpha: float = 1.0):
        """Paste a numpy array (image) onto the canvas"""
        img = self._prepare_image_array(image)
        h, w = img.shape[:2]
        
        if size is None:
            target_w, target_h = w, h
        else:
            target_w, target_h = size
        
        texture = self.ctx.texture((w, h), 4, img.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        x, y = position
        
        vertices = np.array([
            x, y, 0.0, 0.0,
            x + target_w, y, 1.0, 0.0,
            x + target_w, y + target_h, 1.0, 1.0,
            x, y, 0.0, 0.0,
            x + target_w, y + target_h, 1.0, 1.0,
            x, y + target_h, 0.0, 1.0
        ], dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(
            self.renderer.shader_texture_2d,
            vbo,
            'in_position', 'in_texcoord_0'
        )
        
        self.renderer.shader_texture_2d['projection'].write(self.renderer.camera.get_projection_matrix())
        self.renderer.shader_texture_2d['model'].write(self.renderer.transform.matrix)
        self.renderer.shader_texture_2d['texture0'] = 0
        self.renderer.shader_texture_2d['alpha'] = alpha
        
        texture.use(0)
        vao.render(moderngl.TRIANGLES)
        
        vao.release()
        vbo.release()
        texture.release()
    
    def _prepare_image_array(self, image: np.ndarray) -> np.ndarray:
        """Prepare image array for OpenGL texture"""
        img = np.asarray(image)
        
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        if len(img.shape) == 2:
            h, w = img.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = img
            rgba[:, :, 3] = 255
            img = rgba
        elif img.shape[2] == 3:
            h, w = img.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = 255
            img = rgba
        elif img.shape[2] != 4:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        
        return img
    
    def _flush_for_fbo_change(self):
        """Flush batch because FBO is changing"""
        batches = self.renderer.batch_manager.group_commands(self.renderer.draw_queue)
        for batch in batches:
            self.renderer.batch_manager.render_batch(batch)
        
        self.renderer.draw_queue.clear()
        self.renderer.draw_order_counter = 0