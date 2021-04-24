import tifffile as tiff
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
#loading images into one array of dimension (batch_size,width,height,channels)
def load_images(image_path):
    image_list = []
    for image in os.listdir(image_path):
        if image.endswith('.tiff'):
            image = tiff.imread(os.path.join(image_path,image))
            image_list.append(image)
    image_list = np.array(image_list)
    return image_list

def load_masks(mask_path):
    mask_list = []
    for mask in os.listdir(mask_path):
        if mask.endswith('.png'):
            mask = cv2.imread(os.path.join(mask_path,mask),0)
            mask_list.append(mask)
    mask_list = np.array(mask_list)
    return mask_list

def z_norm(arr, epsilon=1e-100):
    return (arr-arr.mean())/(arr.std()+epsilon)  

def normalization(images_array):
    images_array[:,:,:,2] = (images_array[:,:,:,2]/10000) #values are from 0 to 1000, so we can just divide by 1000 to normalize
    for i in range(0,3):
        images_array[:,:,:,i] = z_norm(images_array[:,:,:,i])
    images_array = images_array.astype(np.float32)
    return images_array

def change_mask_classes(masks_array):
    masks_array[masks_array == 254] = 1
    masks_array[masks_array == 127] = 2
    return masks_array

