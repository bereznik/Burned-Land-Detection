import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import confusion_matrix

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
        self.precision = (self.TP)/(self.TP+self.FP + 1e-10)
        self.recall = (self.TP)/(self.TP + self.FN +1e-10)

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
    for i in range(0,len(y_test)):
        conf = confusion_matrix(y_true[i].flatten(),y_pred[i].flatten(),labels=[0,1,2])
        metrics = Metrics(conf,k)
       
        if (metrics.precision == 0)&(metrics.recall ==0):
            continue
        j= j+1
        scores_precision = scores_precision + metrics.precision
        scores_recall = scores_recall + metrics.recall
        scores_accuracy = scores_accuracy + metrics.accuracy
    
    scores = dict({'Accuracy':scores_accuracy/j,'Precision':scores_precision/j,'Recall':scores_recall/j,'F1':2*(scores_precision/j)*(scores_recall/j)/(scores_recall/j + scores_precision/j)})
    return scores

def compare_masks(y_pred,y_true,n):
    fig, ax = plt.subplots(1,2)
    cmap = colors.ListedColormap(['black','#036A14','#27D644'])
    ax[0].imshow(y_pred[n],cmap=cmap)
    ax[1].imshow(y_true[n],cmap=cmap)
    