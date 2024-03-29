import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import tifffile
from ndvi import make_ndvi
import rasterio
np.seterr(divide='ignore', invalid='ignore')
import cv2

def change_names():
    image_path = '../Binary Masks/Cortes/Images'
    mask_path = '../Binary Masks/Cortes/Masks'

    for image in os.listdir(image_path):
        if image.endswith('.xml'):
            os.remove(os.path.join(image_path,image))
    for mask in os.listdir(mask_path):
        if mask.endswith('.xml'):
            os.remove(os.path.join(mask_path,mask))
    i = 1

    for image,mask in zip(os.listdir(image_path),os.listdir(mask_path)):
        os.rename(os.path.join(image_path,image),os.path.join(image_path,'image_' + str(i)))
        os.rename(os.path.join(mask_path,mask),os.path.join(mask_path,'mask_' + str(i)))
        i = i+1



def make_save_subimages(raster_before_path,raster_after_path,im_path_1,im_path_2,out_path,n):
    ''' im_array_path: path to numpy array representing the Geotif Image
        im_path_1: path to binary mask containing class 1 (forest)
        im_path_2: path to binary mask containing class 2 (field)
        n: number of subimages to create
        out_path: path to save output images 

    Generates sub images form binary mask randomly but containing for sure target pixels
    '''
    #loading rasters
    raster_before = np.load(raster_before_path).astype(np.int32)
    raster_after = np.load(raster_after_path).astype(np.int32)
    
    #loading masks
    im1 = plt.imread(im_path_1)
    im2 = plt.imread(im_path_2)
    
    im2[np.where(im2!=0)] = 128 # transforms the pixels of the second binary mask to other value to succesfully sum the masks into one
    
    mask = im1 + im2 #creates one mask from the 2 others

    mask_copy = np.copy(mask) 

    ndvi_before,ndvi_after = make_ndvi(raster_before,raster_after)    
    
    size = 256
    for i in range(0,n):
        ar1,ar2 = np.where(mask!=0)
        index = np.random.choice(ar1,n) # find the pixles of the image where there are burnt areas and chooses n random ones
        c = ar1[index]
        d = ar2[index]
        
        a = np.random.randint(0,size) # generates random numbers to shift the image
        b = np.random.randint(0,size) 
        
        cut_mask = mask_copy[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        
        cut_ndvi_after = ndvi_after[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        cut_ndvi_before = ndvi_before[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        mean_NIR = raster_before[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b,1]/2 + raster_after[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b,1]/2

        final_raster = np.dstack((cut_ndvi_before,cut_ndvi_after,mean_NIR)).astype(np.float32)
        image_cut_mask = Image.fromarray(cut_mask*255)
        if (final_raster== 0).sum() != 0:
            continue
        
        if (os.path.isdir(out_path + '/Images')) & (os.path.isdir(out_path + '/Masks')) == False:
            os.mkdir(out_path + '/Images')
            os.mkdir(out_path + '/Masks')
        
      

        tifffile.imsave(out_path+'/Images'+'/image_'+str(i)+'.tiff',final_raster,planarconfig = 'contig')
        image_cut_mask.save(out_path+'/Masks'+'/mask_' + str(i)+'.png')
        
        # Now, to prevent to get the same image a lot of times, we take out
        # the alredy used pixels.
        used_pixels_sub_array = np.array(np.where(cut_mask!=0))
        used_pixels_array_x = used_pixels_sub_array[0] + c[i] - a
        used_pixels_array_y = used_pixels_sub_array[1] + d[i] - b
        mask[used_pixels_array_x,used_pixels_array_y] = 0
    
    #change_names()

def reshape_split(image,kernel_size=(256,256)):
    img_height,img_width, channels = image.shape
    tile_height,tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1,2)
    return tiled_array

def concatenate_splited_image(image):
    n_lines = image.shape[0]
    n_columns = image.shape[1]
    horizontal_concat = []
    for i in range(0,n_lines):
        horizontal_concat.append(cv2.hconcat([image[i,x] for x in range(0,n_columns)]))
    
    final_image = cv2.vconcat([img for img in horizontal_concat])
    return final_image



