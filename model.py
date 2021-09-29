import tifffile as tiff
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
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

def z_norm(arr, epsilon=1e-10):
    return (arr-arr.mean())/(arr.std()+epsilon)  

def normalization(images_array,mode):
    
    if mode == 'train':
        images_array[:,:,:,2] = (images_array[:,:,:,2]/10000) #values are from 0 to 1000, so we can just divide by 1000 to normalize
        for i in range(0,3):
            images_array[:,:,:,i] = z_norm(images_array[:,:,:,i])
        images_array = images_array.astype(np.float32)
    
    if mode == 'generalization':
        images_array[:,:,2] = images_array[:,:,2]
        for i in range(0,3):
            images_array[:,:,i] = z_norm(images_array[:,:,i])
        images_array = images_array.astype(np.float32)
    return images_array

def change_mask_classes(masks_array):
    masks_array[masks_array == 254] = 1
    masks_array[masks_array == 127] = 2
    masks_array[masks_array == 125] = 2
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

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = keras.reshape(y_pred, (-1, keras.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = keras.one_hot(tf.to_int32(keras.flatten(y_true).astype), keras.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -keras.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = keras.mean(cross_entropy)

    return cross_entropy_mean

def multi_unet_model(n_classes=3, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    model.compile(optimizer='adam', loss='categorical_crossentropy'
    , metrics=['accuracy'])
    
    print(model.summary())
    
    return model



    