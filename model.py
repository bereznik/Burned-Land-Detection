import tifffile as tiff
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
#loading images into one array of dimension (batch_size,width,height,channels)
def load_images(image_path):
    image_list = []
    for image in sorted(os.listdir(image_path)):
        if image.endswith('.tiff'):
            image = tiff.imread(os.path.join(image_path,image))
            image_list.append(image)
    image_list = np.array(image_list)
    return image_list

def load_masks(mask_path):
    mask_list = []
    for mask in sorted(os.listdir(mask_path)):
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

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

def data_augmentation(batch_images,batch_masks):
    images = []
    masks = []
    for image,mask in zip(batch_images,batch_masks):
        #Rotations
        for i in range(0,4):
            transform_image,transform_mask = A.RandomRotate90(p=1).apply(image,factor=i),A.RandomRotate90(p=1).apply(mask,factor=i) 
            images.append(transform_image)
            masks.append(transform_mask)
        #Horizontal Flip
        transform_image,transform_mask = A.HorizontalFlip(p=1).apply(image),A.HorizontalFlip(p=1).apply(mask)
        images.append(transform_image)
        masks.append(transform_mask)

        #Vertical Flip
        transform_image,transform_mask = A.VerticalFlip(p=1).apply(image),A.VerticalFlip(p=1).apply(mask)
        images.append(transform_image)
        masks.append(transform_mask)

        #Diagonal flip 1
        transform_image,transform_mask = A.Transpose(p=1).apply(image),A.Transpose(p=1).apply(mask) 
        images.append(transform_image)
        masks.append(transform_mask)

        #Diagonal flip 2
        transform_image,transform_mask = A.RandomRotate90(p=1).apply(image,factor=2),A.RandomRotate90(p=1).apply(mask,factor=2)
        transform_image,transform_mask = A.Transpose(p=1).apply(transform_image),A.Transpose(p=1).apply(transform_mask)
        images.append(transform_image)
        masks.append(transform_mask)
    
    all_images = np.array(images)
    all_masks = np.array(masks)
    
    
    return all_images,all_masks               



