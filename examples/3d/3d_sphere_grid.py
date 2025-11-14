from dorothy import Dorothy
import math
dot = Dorothy(1080,960)

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)           
        
    def setup(self):
        #Listen to mic or internal loop back (e.g. blackhole)
        dot.music.start_device_stream(2)
        dot.camera_3d()
        self.theta = 0

    def draw(self):
        dot.background(dot.white)
        dot.fill(dot.red)
        amp = dot.music.amplitude()
        self.theta += amp
        x = 7 * math.cos(self.theta)
        z = 7 * math.sin(self.theta)
        dot.set_camera((x, 3, z), (0, 0, 0))
        
        # Grid centered at origin
        for i in range(10):
            for j in range(10):
                x = (i - 5) * 1  # -250 to 250
                y = (j - 5) * 1
                z = 0
                #dot.box((amp * 20,amp * 20,amp * 20),(x, y, z))
                dot.sphere(amp * 20,(x, y, z))
                # print(x,y)
        dot.rgb_split(accumulate=False, offset=amp*0.5)

MySketch()   
    







