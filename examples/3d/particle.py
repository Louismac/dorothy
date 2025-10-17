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
        
        self.size = 1 +np.random.random() * 10
        self.life = 1
        self.decay = np.random.random() * 0.01 + 0.003
    
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
        dot.fill((255*self.life,255,0,255*self.life))
        
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
        dot.camera_3d()
        self.particles = []
        self.emit_position = [0, 0, 0]
        self.scale_lfo = dot.get_lfo(freq = 0.1, range = (50,90))
        self.spin_lfo = dot.get_lfo(freq = 0.1, range = (0, np.pi))
        self.camera_angle = 0
    
    def draw(self):
        dot.background(dot.black)
        z_pos = 1 + dot.lfo_value(self.scale_lfo)
        x_pos = dot.lfo_value(self.spin_lfo)
        # Rotating camera
        self.camera_angle += 0.01
        radius = 100
        x_pos = np.cos(self.camera_angle) * radius
        y_pos = 0
        dot.set_camera((x_pos,y_pos,z_pos),(0,0,0))
        
        # Emit particles continuously
        for _ in range(1):
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

MySketch()