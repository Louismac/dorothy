from dorothy import Dorothy
from skimage import measure #(pip install scikit-image)
import numpy as np
import cv2
import sounddevice as sd

dot = Dorothy(1920,1080)

class MySketch:

    image = []
    bbox = []
    mode = 0
    written = False
    level = 0.4
    topk = 100

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        
        fp = "../images/aligned_faces/face_68.jpg"

        self.image = cv2.imread(fp)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.im_w = self.image.shape[1]
        self.im_h = self.image.shape[0]
        print(self.im_w,self.im_h)

        if self.im_h > dot.height or self.im_w > dot.width:
            print("scaling down image to fit canvas")
            self.image = cv2.resize(self.image, (dot.width, dot.height))
            self.im_w = self.image.shape[1]
            self.im_h = self.image.shape[0]
            print(self.im_w,self.im_h)

        self.gray_example = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, self.thresh= cv2.threshold(self.gray_example, 120, 255, cv2.THRESH_BINARY)
        self.thresh +=1
        
        #Get contours 
        self.contours = measure.find_contours(self.thresh, self.level)
        self.contours = [np.array(c[:, [1, 0]], dtype=np.int32) for c in self.contours if len(c)>3]
        
        #Filter contours 
        new_list = []
        
        #Only keep closed
        # self.contours= [c for c in self.contours if c[0,1] == c[-1,1] and c[0,0] == c[-1,0]]
        #Or close open ones
        for c in self.contours:
            if not (c[0,1] == c[-1,1] and c[0,0] == c[-1,0]):
                c[-1] = c[0]
                
        self.areas = np.array([cv2.contourArea(contour) for contour in self.contours])

        #Keep top k biggest contours (by area)
        indexes = self.areas.argsort()[-self.topk:]
        for i in indexes:
            if self.areas[i]>10:
                contour = self.contours[i]
                new_list.append(contour)
                self.bbox.append(cv2.boundingRect(contour))

        self.contours = new_list
        print("Contours:", len(self.contours))
        
        self.top_y = (dot.height-self.im_h)//2
        self.top_x = (dot.width-self.im_w)//2
        self.theta_offset = np.pi

        

        #Pick or just stream from your computer
        #On MacOSX I use Blackhole and Multioutput device to pump audio to here, and to listen in speakers as well
        print(sd.query_devices())
        dot.music.start_file_stream("../audio/hiphop.wav")

        dot.background((0,0,0))
        #Draw image behind?
        #dot.paste(dot.canvas, self.image, (self.top_x, self.top_y))
        self.color = [0,0,0]  
        #interpolate between these colours
        self.a = dot.cyan
        self.b = dot.magenta
        self.face_layer = dot.get_layer()

    def draw(self):
        t = (np.sin(np.pi*(dot.frame/1000))+1)/2
        for i in range(3):
            self.color[i] = int(self.a[i] + (self.b[i] - self.a[i]) * t)
        # dot.background(self.color)
        #Draw image behind?
        #dot.paste(dot.canvas, self.image, (self.top_x, self.top_y))
        
        #Transparency of contours
        alpha = 0.6
        #Chance of a direction change
        dir_change = 0.05
        #How many to draw
        cut_off = 50
        #Scale of movement
        scale = 2

        #Get contours to draw
        to_use = self.contours[-cut_off:]
        filtered_bbox = self.bbox[-cut_off:]
        
        fft_vals = dot.music.fft(0)
        
        #Should we change direction?
        if np.random.random()<dir_change:
            self.theta_offset = np.random.choice([np.pi * i for i in np.linspace(0,2,16)])
        
        #Iterate through
        for index, contour in enumerate(to_use):
            #cut out the shape
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            cut_out = cv2.bitwise_and(self.image, self.image, mask=mask)
            x,y,w,h = filtered_bbox[index]
            cut_out = cut_out[y:y+h,x:x+w]
            cut_out += 1
            cut_out = np.array(cut_out, np.uint8)

            #Get angle
            theta = (np.pi*2) * (index/len(to_use))            
            theta += self.theta_offset

            #Get coordinates
            delta_y = np.abs((y + self.top_y) - dot.height//2)
            delta_x = np.abs((x + self.top_x) - dot.width//2)
            r = np.sqrt((delta_x**2 + delta_y**2)) 
            r = (r * (fft_vals[index])/6)
            r *= scale
            
            new_y = int((y + self.top_y) + ((r)*np.cos(theta)))
            new_x = int((x + self.top_x) + ((r)*np.sin(theta)))
            
            dot.paste(self.face_layer, cut_out, (new_x, new_y))

        dot.draw_layer(self.face_layer)

MySketch()          
