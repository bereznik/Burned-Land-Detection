import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os
import tifffile

def make_save_subimages(image_array_path,im_path_1,im_path_2,out_path,n):
    ''' im_array_path: path to numpy array representing the Geotif Image
        im_path_1: path to binary mask containing class 1 (forest)
        im_path_2: path to binary mask containing class 2 (field)
        n: number of subimages to create
        out_path: path to save output images 

    Generates sub images form binary mask randomly but containing for sure target pixels
    '''
    raster = np.load(image_array_path) #reads numpy array from memory
    im1 = plt.imread(im_path_1) #reads binary masks as numpy array
    im2 = plt.imread(im_path_2)
    
    im2[np.where(im2!=0)] = 128 # transforms the pixels of the second binary mask to other value to succesfully sum the masks into one
    
    mask = im1 + im2 #creates one mask from the 2 others
    
    ar1,ar2 = np.where(mask!=0)
    index = np.random.choice(ar1,n) # find the pixles of the image where there are burnt areas and chooses n random ones
    c = ar1[index]
    d = ar2[index]
    
    size = 256
    for i in range(0,n):
        a = np.random.randint(0,size) # generates random numbers to shift the image
        b = np.random.randint(0,size) 
        cut_mask = mask[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        cut_raster = raster[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b,0:4]
        image_cut_mask = Image.fromarray(cut_mask*255)
        
        if (cut_raster == 0).sum() != 0:
            continue
        
        if os.path.isdir(out_path + str(i)) == False:
            os.mkdir(out_path + str(i))
            tifffile.imsave(out_path+str(i)+'/Raster_'+str(i)+'.tiff',cut_raster,planarconfig = 'contig')
            image_cut_mask.save(out_path+str(i)+'/Mask_' + str(i)+'.png')
        else:
            tifffile.imsave(out_path+str(i)+'/Raster_'+str(i)+'.tiff',cut_raster,planarconfig = 'contig')
            image_cut_mask.save(out_path+str(i)+'/Mask_' + str(i)+'.png')