import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import tifffile
from ndvi import make_ndvi
np.seterr(divide='ignore', invalid='ignore')
def make_save_subimages(raster_before_path,raster_after_path,im_path_1,im_path_2,out_path,n):
    ''' im_array_path: path to numpy array representing the Geotif Image
        im_path_1: path to binary mask containing class 1 (forest)
        im_path_2: path to binary mask containing class 2 (field)
        n: number of subimages to create
        out_path: path to save output images 

    Generates sub images form binary mask randomly but containing for sure target pixels
    '''
    #loading rasters
    raster_before = np.load(raster_before_path)
    raster_after = np.load(raster_after_path)
    
    #loading masks
    im1 = plt.imread(im_path_1)
    im2 = plt.imread(im_path_2)
    
    im2[np.where(im2!=0)] = 128 # transforms the pixels of the second binary mask to other value to succesfully sum the masks into one
    
    mask = im1 + im2 #creates one mask from the 2 others
    
    ar1,ar2 = np.where(mask!=0)
    index = np.random.choice(ar1,n) # find the pixles of the image where there are burnt areas and chooses n random ones
    c = ar1[index]
    d = ar2[index]

    ndvi_before,ndvi_after = make_ndvi(raster_before,raster_after)    

    size = 256
    for i in range(0,n):
        a = np.random.randint(0,size) # generates random numbers to shift the image
        b = np.random.randint(0,size) 
        
        cut_mask = mask[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        
        cut_ndvi_after = ndvi_after[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        cut_ndvi_before = ndvi_before[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b]
        mean_NIR = (raster_before[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b,1] + raster_after[c[i]-a:c[i]+size-a,d[i]-b:d[i]+size-b,1])/2

        final_raster = np.dstack((cut_ndvi_before,cut_ndvi_after,mean_NIR))
        image_cut_mask = Image.fromarray(cut_mask*255)
        if (cut_ndvi_after== 0).sum() != 0:
            continue
        
        if os.path.isdir(out_path + str(i)) == False:
            os.mkdir(out_path + str(i))
            tifffile.imsave(out_path+str(i)+'/Raster_'+str(i)+'.tiff',final_raster,planarconfig = 'contig')
            image_cut_mask.save(out_path+str(i)+'/Mask_' + str(i)+'.png')
        else:
            #tifffile.imsave(out_path+str(i)+'/Raster_'+str(i)+'.tiff',cut_raster_after,planarconfig = 'contig')
            image_cut_mask.save(out_path+str(i)+'/Mask_' + str(i)+'.png')
            ndvi_cut.save(out_path+str(i)+'/Raster_'+str(i)+'.png') 