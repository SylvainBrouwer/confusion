import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable, ALL
from ._base import BaseConfusionMatrix, _convert_to_bool_np


class ConfusionMatrix(BaseConfusionMatrix):

    def __init__(self, y_true, y_pred, classnames=("P", "N")):
        y_true = _convert_to_bool_np(y_true)
        y_pred = _convert_to_bool_np(y_pred)
        
        if y_true.shape != y_pred.shape or y_true.ndim != 1:
            raise ValueError("Inputs must be 1d and have the same shape.")
        if len(classnames) != 2:
            raise ValueError("Two classnames should be provided.")
        
        self.classnames = classnames
        self._TP_map = np.logical_and(y_true, y_pred)
        self._TN_map = np.logical_not(np.logical_or(y_true, y_pred))
        self._FP_map = np.logical_and(np.logical_not(y_true), y_pred)
        self._FN_map = np.logical_and(y_true, np.logical_not(y_pred))


    # Properties
    @property
    def TP(self):
        return sum(self._TP_map)
    
    @property
    def TN(self):
        return sum(self._TN_map)
    
    @property
    def FP(self):
        return sum(self._FP_map)
    
    @property
    def FN(self):
        return sum(self._FN_map)
    
    
    # Base metrics
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

    # Dependent metrics
    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)

    def f_beta(self, beta):
        if self.TP == 0:
            return 0
        return (1+pow(beta, 2))*((self.precision()*self.recall())/(pow(beta, 2)*self.precision()+self.recall()))
    
    def FDR(self):
        return 1 - self.precision()
    
    def FNR(self):
        return 1 - self.recall()

    def FOR(self):
        return 1 - self.NPV()
    
    def FPR(self):
        return 1 - self.specificity()
    
    # Printing
    def __repr__(self) -> str:
        table = PrettyTable(header=False, border=True, hrules=ALL)
        table.add_row(["", f"{self.classnames[0]} true", f"{self.classnames[1]} true"])
        table.add_row([f"{self.classnames[0]} predicted", self.TP, self.FP])
        table.add_row([f"{self.classnames[1]} predicted", self.FN, self.TN])
        return "\n"+table.get_string()
    
    # Plotting
    def plot(self):
        table_T = np.array([[self.TP, 0], [0, self.TN]])
        table_F = np.array([[0, self.FP], [self.FN, 0]])
        cmap_T = plt.get_cmap("Greens")
        cmap_F = plt.get_cmap("Reds")
        mask =  np.array([[True, False],[False, True]])
        ax = sns.heatmap(
            data=table_T,
            vmin=0, 
            annot=True, 
            cmap=cmap_T,
            cbar=False,
            linewidth=1, 
            )
        sns.heatmap(
            data=table_F,
            vmin=0,
            annot=True,
            cmap=cmap_F,
            cbar=False,
            mask=mask,
            linewidth=1,
            ax=ax
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_ticklabels(labels=self.classnames)
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_ticklabels(labels=self.classnames, rotation="horizontal")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        plt.show()