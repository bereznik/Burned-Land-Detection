from generate_subimages import concatenate_splited_image, reshape_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
from keras.models import *
import rasterio
from ndvi import make_ndvi
from model import normalization
import cv2

class Metrics:
    '''
        Return Metrics of confusion matrix for each class k
    '''
    def __init__(self,conf,k):
        self.TP = conf[k,k]
        
        sum = 0
        for i in range(0,len(conf)):
            if i == k:
                continue
            for j in range(0,len(conf)):
                if j == k:
                    continue
                sum = sum + conf[i,j]
        self.TN = sum

        sum = 0
        for i in range(0,len(conf)):
            if i == k:
                continue
            sum = sum + conf[i,k]
        self.FP = sum

        sum = 0
        for j in range(0,len(conf)):
            if j == k:
                continue
            sum = sum + conf[k,j]
        self.FN = sum

        self.accuracy = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN + 1e-10)
        self.precision = (self.TP)/(self.TP+self.FP+1e-10)
        self.recall = (self.TP)/(self.TP + self.FN +1e-10)

    def print_metrics(self):
        print("TP = ",self.TP)
        print("FP = ",self.FP)

def model_evaluation(model,x_test,y_test,k):
    '''
    Evaluates the current model
    -------
    params: model -> an instance of a Model object representing the network
            x_test -> validation input data
            y_test -> validation output data
            k -> class to output predictions (0 = not burned land, 1 = forest burned land, 2 = pasture b
            urned land)
    '''
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred,axis=3)
    y_true = np.argmax(y_test, axis = 3)
    j = 0
    scores_accuracy = 0
    scores_precision = 0
    scores_recall = 0
    confi = confusion_matrix(y_true.flatten(),y_pred.flatten(),labels=[0,1,2])
    metricsi = Metrics(confi,k)
    # for i in range(0,len(y_test)):
    #     conf = confusion_matrix(y_true[i].flatten(),y_pred[i].flatten(),labels=[0,1,2])
    #     metrics = Metrics(conf,k)
       
    #     if (metrics.precision == 0)|(metrics.recall ==0):
    #         continue
    #     j= j+1
    #     scores_precision = scores_precision + metrics.precision
    #     scores_recall = scores_recall + metrics.recall
    #     scores_accuracy = scores_accuracy + metrics.accuracy
    scores_precision = metricsi.precision
    scores_recall = metricsi.recall
    scores_accuracy = metricsi.accuracy

    # scores = dict({'Accuracy':scores_accuracy/j,'Precision':scores_precision/j,'Recall':scores_recall/j,'F1':2*(scores_precision/j)*(scores_recall/j)/(scores_recall/j + scores_precision/j)})
    scores = dict({'Accuracy':scores_accuracy,'Precision':scores_precision,'Recall':scores_recall,'F1':2*(scores_precision)*(scores_recall)/(scores_recall + scores_precision)})
    return scores

def compare_masks(y_pred,y_true,n):
    fig, ax = plt.subplots(1,2)
    cmap = colors.ListedColormap(['black','#036A14','#27D644'])
    ax[0].imshow(y_pred[n],cmap=cmap)
    ax[1].imshow(y_true[n],cmap=cmap)
    # Hide grid lines
    
    ax[0].grid(False)
    ax[1].grid(False)


    # Hide axes ticks
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])

def save_masks(y_pred,y_true,n):
    cmap = colors.ListedColormap(['black','#036A14','#27D644'])
    path = '../Relatório/Imagens/Resultados/Máscara ' +str(n)
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.imsave(path + '/Real.png',y_true[n],cmap =cmap)
    plt.imsave(path+'/Predito.png',y_pred[n],cmap =cmap)

#-------------------------------- evaluate model on a new image to check generalization


def get_array_from_raster(path_raster):
    array = rasterio.open(path_raster).read((3,4)) #returns an array with RED and NIR bands
    array=array.swapaxes(0,2).swapaxes(1,0).astype(np.int32)
    return array

def get_final_image_from_raster(path_raster_before,path_raster_after):
    array_before = get_array_from_raster(path_raster_before)
    array_after = get_array_from_raster(path_raster_after)
    ndvi_before,ndvi_after = make_ndvi(array_before,array_after)
    mean_NIR = (array_before[:,:,1] + array_after[:,:,1])/2
    print('ndvi shape ->', ndvi_before.shape, 'mean_NIR shape ->',mean_NIR.shape)
    final_image = np.dstack((ndvi_before,ndvi_after,mean_NIR))
    return final_image

def convert_NIR_band_to_gray(image):
    img = image[:,:,2]
    new_img = np.zeros(img.shape).astype(np.uint8)
    new_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return new_img


def get_rotated_croped_final_image_from_raster(path_raster_before,path_raster_after):
    
    final_image = get_final_image_from_raster(path_raster_before,path_raster_after)
    gray = convert_NIR_band_to_gray(final_image)
    # binarize image
    threshold, binarized_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find the largest contour
    contours, hierarchy = cv2.findContours(binarized_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)

    # get size of the rotated rectangle
    center, size, angle = cv2.minAreaRect(c)

    # get size of the image
    h, w, *_ = final_image.shape

    # create a rotation matrix and rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated_img = cv2.warpAffine(final_image, M, (w, h))

    # crop the image
    pad_x = int((w - size[0]) / 2)
    pad_y = int((h - size[1]) / 2)

    cropped_img = rotated_img[pad_y : pad_y + int(size[1]), pad_x : pad_x + int(size[0])]#, :]

    return cropped_img


def get_right_dimensions_to_split(image,size = 256): #dimensions need to be multiples of size
    n_lines = (image.shape[0]//size)*size
    n_columns = (image.shape[1]//size)*size
    return (n_lines,n_columns)

def get_image_to_split_with_right_dimensions(image):
    n_lines,n_columns = get_right_dimensions_to_split(image)
    return image[0:n_lines,0:n_columns,:]

def get_normalized_image_to_split(image):
    im = get_image_to_split_with_right_dimensions(image)
    im = normalization(im,mode = 'generalization')
    return im

def get_splited_image(image):
    return reshape_split(get_normalized_image_to_split(image))


def get_predictions_on_splited_images(image,model_parameters = 'model_parameters/params'):
    model = tf.keras.models.load_model(model_parameters)
    splited_image = get_splited_image(image)
    predicted_splited_images = []
    for img in splited_image:
        predicted_img = model.predict(img)
        predicted_img = np.argmax(predicted_img,axis = 3)
        predicted_splited_images.append(predicted_img)
    predicted_splited_images = np.array(predicted_splited_images)
    return predicted_splited_images

def get_whole_prediction(image):
    predictions = get_predictions_on_splited_images(image)
    whole_image = concatenate_splited_image(predictions)
    return whole_image
