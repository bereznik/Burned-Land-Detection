import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.precision = (self.TP)/(self.TP+self.FP + 1e-6)
        self.recall = (self.TP)/(self.TP + self.FN +1e-6)

def model_evaluation(model,x_test,y_test,k):
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
