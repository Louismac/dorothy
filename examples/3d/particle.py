from dorothy import Dorothy
import numpy as np

dot = Dorothy(800, 800)

class Particle3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        # Random velocity in 3D
        angle_xz = np.random.random() * np.pi * 2
        angle_y = (np.random.random() - 0.5) * np.pi
        speed = 0.3
        
        self.vx = np.cos(angle_xz) * np.cos(angle_y) * speed
        self.vy = np.sin(angle_y) * speed
        self.vz = np.sin(angle_xz) * np.cos(angle_y) * speed
        
        self.size = 1 +np.random.random() * 20
        self.life = 1
        self.decay = np.random.random() * 0.005 + 0.001
    
    def update(self):
        # Apply velocity
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        
        # Fade out
        self.life -= self.decay
        
    def is_alive(self):
        return self.life > 0
    
    def draw(self):
  
        # Set color with life-based alpha
        dot.fill((255*self.life,0,255,255*self.life))
        
        # Position and draw sphere
        with dot.transform():
            # print(self.x, self.y, self.z)
            # print(self.vx, self.vy, self.vz)
            dot.translate(self.x, self.y, self.z)
            dot.sphere(2 + self.size*self.life*dot.music.amplitude()*5)

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)
    
    def setup(self):
        dot.music.start_file_stream("../audio/hiphop.wav")
        dot.background(dot.red)
        dot.camera_3d()
        self.particles = []
        self.emit_position = [0, 0, 0]
        self.scale_lfo = dot.get_lfo(freq = 0.1, range = (50,90))
        self.spin_lfo = dot.get_lfo(freq = 0.1, range = (0, 1))
        self.camera_angle = 0

        self.emit_shader = '''
        #version 330
        uniform sampler2D texture0;
        uniform float glow_boost;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec4 color = texture(texture0, v_texcoord);
            
            // Boost bright colors
            float brightness = length(color.rgb);
            if (brightness > 0.5) {
                color.rgb *= (1.0 + glow_boost);
            }
            
            fragColor = color;
        }
        '''
        
        self.fast_blur = '''
        #version 330
        uniform sampler2D texture0;
        uniform vec2 resolution;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec2 pixel = 1.0 / resolution;
            vec4 color = vec4(0.0);
            
            // 3x3 blur
            for(int x = -1; x <= 1; x++) {
                for(int y = -1; y <= 1; y++) {
                    color += texture(texture0, v_texcoord + vec2(x, y) * pixel);
                }
            }
            
            fragColor = color / 9.0;
        }
        '''
    
    def draw(self):
        dot.background((0,0,0,255))
        radius = 100
        # dot.lfos[self.spin_lfo]["freq"] = dot.music.amplitude()*1
        z_pos = 1 + dot.lfo_value(self.scale_lfo)
        x_pos = dot.lfo_value(self.spin_lfo)*radius
        y_pos = 0
        dot.set_camera((x_pos,y_pos,z_pos),(0,0,0))
        
        # Emit particles continuously
        if np.random.random()>0.75:
            px = 10-(dot.mouse_x/dot.width)*20
            py = 10-(dot.mouse_y/dot.height)*20
            pz = self.emit_position[2]
            self.particles.append(Particle3D(px,py,pz))
        
        # Update and draw particles
        alive_particles = []
        for p in self.particles:
            p.update()
            if p.is_alive():
                p.draw()
                alive_particles.append(p)
        self.particles = alive_particles

        # Boost emission
        # dot.apply_shader(self.emit_shader, accumulate=True, glow_boost=0.1)
        # dot.background((0,0,0,4))
        # # Blur multiple times for softer glow
        # for _ in range(3):
        #     dot.apply_shader(self.fast_blur, accumulate=True)

MySketch()