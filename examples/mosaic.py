import numpy as np
from PIL import Image
from scipy import spatial
import PIL
from dorothy import Dorothy

dot = Dorothy()

src_image_path = '/Users/louisbusby/Downloads/dogcutout.jpg'
dataset_path = "/Users/louisbusby/Downloads/franks"
thumbnail_size = (15,15)
downsample_rate = 15
k = 20

#Convert PIL image object to numpy array
target_im_np = np.array(Image.open(src_image_path))

#Downsample image using the downsample rate parameter
mosaic_template = np.swapaxes(target_im_np[::downsample_rate,::downsample_rate],0,1)

dataset = dot.get_images(dataset_path, thumbnail_size = thumbnail_size)
#Find the mean of each colour channel for each image in the dataset
image_values = np.apply_over_axes(np.mean, dataset, [1,2]).reshape(dataset.shape[0],3)

#Calculate a binary search tree which is used to efficiently assign image thumbnails to pixel values (based on the closest colour mactch)
tree = spatial.KDTree(image_values)

#Variables to store which image is assigned to which pixel
target_res = mosaic_template.shape[0:2]
image_idx = np.zeros(target_res, dtype=np.uint32)

#Go through each pixel and find the closest matching thumbnail image by mean colour and assign the index into the 2D array
for i in range(target_res[0]):
    for j in range(target_res[1]):
        template = mosaic_template[i, j]
        match = tree.query(template, k=k)
        pick = np.random.randint(k)
        image_idx[i, j] = match[1][pick]

#Variable that can contail all the pixel values for the new image
mosaic = PIL.Image.new('RGB', (thumbnail_size[0]*target_res[0], thumbnail_size[1]*target_res[1]))

#Go through each pixel in the array of thumbnail<>pixel indexes and then assign all the pixels of the thumbnail into the final array
for i in range(target_res[0]):
    for j in range(target_res[1]):
        arr = dataset[image_idx[i, j]]
        #Calculate the coordinate of where to paste in mosaic
        x, y = i * thumbnail_size[0], j * thumbnail_size[1]
        im = PIL.Image.fromarray(arr)
        mosaic.paste(im, (x,y))
    
#Save the photomosaic to a file
mosaic.save('mosaic.png')