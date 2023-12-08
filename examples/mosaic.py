import glob 
import PIL
import numpy as np
from matplotlib.image import thumbnail
from skimage.transform import resize
from PIL import ImageOps
from scipy import spatial

def get_images(root_dir = "data/animal_thumbnails/land_mammals/cat", thumbnail_size = (50,50)):
    #Set the thumbnail size (you can change this but you won't want to make it too big!)
    images = []
    #Search through the separate file extensions 
    for ext in ('*.jpeg', '*.jpg', '*.png'):
        #Search through all the image files recursively in the directory
        for file in glob.glob(f'{root_dir}/**/{ext}', recursive=True):
            #Open the image using the file path
            with PIL.Image.open(file) as im:
                #Create a downsampled image based on the thumbnail size
                thumbnail = ImageOps.fit(im, thumbnail_size)
                thumbnail = np.asarray(thumbnail)
                #Check not grayscale (only has 2 dimensions)
                if len(thumbnail.shape) == 3:
                    #Append thumbnail to the list of all the images
                    #Drop any channels beyond rbg (e.g. Alpha for .png files)
                    images.append(thumbnail[:,:,:3])

    print(f'There have been {len(images)} images found')
    #Convert list of images to a numpy array
    image_set_array = np.asarray(images)
    return image_set_array

def generate_image_collage(target_image, image_set_array, downsample_rate, thumbnail_size, k = 40):
    #Convert PIL image object to numpy array
    target_im_np = np.asarray(target_image)
    
    #Downsample image using the downsample rate parameter
    mosaic_template = np.swapaxes(target_im_np[::downsample_rate,::downsample_rate],0,1)
    
    #Find the mean of each colour channel for each image in the dataset
    image_values = np.apply_over_axes(np.mean, image_set_array, [1,2]).reshape(image_set_array.shape[0],3)
    
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
            pick = np.random.randint(0, k=1)
            image_idx[i, j] = match[1][pick]
    
    #Variable that can contail all the pixel values for the new image
    mosaic = PIL.Image.new('RGB', (thumbnail_size[0]*target_res[0], thumbnail_size[1]*target_res[1]))
    
    #Go through each pixel in the array of thumbnail<>pixel indexes and then assign all the pixels of the thumbnail into the final array
    for i in range(target_res[0]):
        for j in range(target_res[1]):
            arr = image_set_array[image_idx[i, j]]
            #Calculate the coordinate of where to paste in mosaic
            x, y = i * thumbnail_size[0], j * thumbnail_size[1]
            im = PIL.Image.fromarray(arr)
            mosaic.paste(im, (x,y))
    
    #Return the mosaic image
    return mosaic