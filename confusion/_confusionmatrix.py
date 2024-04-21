import numpy as np

from ._base import BaseConfusionMatrix
 
class ConfusionMatrix(BaseConfusionMatrix):

    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays must have the same shape.")
        self.y_true = np.array(y_true, dtype=bool)
        self.y_pred = np.array(y_pred, dtype=bool)
        self.TP_map = np.logical_and(self.y_true, self.y_pred)
        self.TN_map = np.logical_not(np.logical_or(self.y_true, self.y_pred))
        self.FP_map = np.logical_and(np.logical_not(self.y_true), self.y_pred)
        self.FN_map = np.logical_and(self.y_true, np.logical_not(self.y_pred))
        self.TP = sum(self.TP_map)
        self.TN = sum(self.TN_map)
        self.FP = sum(self.FP_map)
        self.FN = sum(self.FN_map)
        

    def precision(self):
        if self.TP+self.FP==0:
            return 0
        return self.TP/(self.TP+self.FP)
    
    def recall(self):
        return self.TP/(self.TP+self.FN)

    def specificity(self):
        return self.TN/(self.TN+self.FP)

    def NPV(self):
        return self.TN/(self.TN+self.FN)

    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)

    def f_beta(self, beta):
        if self.TP == 0:
            return 0
        return (1+beta*beta)*((self.precision()*self.recall())/(beta*beta*self.precision()+self.recall()))
    
    def f_1(self):
        return self.f_beta(1)

    def print_results(self):
        print("Precision:", self.precision())
        print("Recall:", self.recall())
        print("F_1:", self.f_beta(1))
        print("Accuracy:", self.accuracy())



def multilabel_confusion_matrices(y_true:np.ndarray, y_pred:np.ndarray):
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    columns = y_true.shape[1]
    conf_matrices = []
    for col in range(columns):
        conf = ConfusionMatrix(y_true[:, col], y_pred[:, col])
        conf_matrices.append(conf)
    return conf_matrices
    
def print_avg_tuple(avg):
    print("Precision: ", avg[0])
    print("Recall: ", avg[1])
    print("F_1: ", avg[2])
    print("Accuracy: ", avg[3])