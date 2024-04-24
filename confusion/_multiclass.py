import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable, ALL
from ._base import BaseConfusionMatrix, _convert_to_bool_np


class MultiClassConfusionMatrix(BaseConfusionMatrix):

    # TODO: Make classnames optional
    def __init__(self, y_true, y_pred, classnames=None):
        y_true = _convert_to_bool_np(y_true)
        y_pred = _convert_to_bool_np(y_pred)
        self.nclasses = y_true.shape[1]

        self.classnames = classnames
        if self.classnames == None:
            self.classnames = list(range(self.nclasses))

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape.")
        if (y_true.sum(axis=1) != 1).any() or (y_pred.sum(axis=1) != 1).any():
            raise ValueError("MultiClassConfusionMatrix does not support multi-label inputs.")
        if len(classnames) != self.nclasses:
            raise ValueError("Number of provided class names does not match number of classes.")
        
        self.matrix = np.zeros((self.nclasses, self.nclasses), dtype=int)
        trues = np.where(y_true == 1)[1]
        for true, pred in zip(trues, y_pred):
            self.matrix[true] += pred


    # Properties
    def TP(self, cls=None):
        if cls is not None:
            return self.matrix[cls, cls]
        return np.diagonal(self.matrix)
    

    def TN(self, cls=None):
        if cls is not None:
            return np.sum(self.matrix) - self.matrix[:, cls].sum() - self.matrix[cls, :].sum() + self.TP(cls=cls)
        return np.array([self.TN(cls=c) for c in range(self.nclasses)])
 

    def FP(self, cls=None):
        if cls is not None:
            return self.matrix[:, cls].sum() - self.TP(cls=cls)
        return np.array([self.FP(cls=c) for c in range(self.nclasses)])    


    def FN(self, cls=None):
        if cls is not None:
            return self.matrix[cls, :].sum() - self.TP(cls=cls)
        return np.array([self.FN(cls=c) for c in range(self.nclasses)])
    
    
    # Base metrics
    def precision(self, cls=None):
        if cls is not None:
            denominator = self.TP(cls=cls) + self.FP(cls=cls)
            if denominator == 0:
                return None
            return self.TP(cls=cls) / denominator
        return np.array([self.precision(cls=c) for c in range(self.nclasses)], dtype=float)


    def recall(self, cls=None):
        if cls is not None:
            denominator = self.TP(cls=cls) + self.FN(cls=cls)
            if denominator == 0:
                return None
            return self.TP(cls=cls) / denominator
        return np.array([self.recall(cls=c) for c in range(self.nclasses)], dtype=float)


    def specificity(self, cls=None):
        if cls is not None:
            denominator = self.TN(cls=cls) + self.FP(cls=cls)
            if denominator == 0:
                return None
            return self.TN(cls=cls) / denominator
        return np.array([self.specificity(cls=c) for c in range(self.nclasses)], dtype=float)


    def NPV(self, cls=None):
        if cls is not None:
            denominator = self.TN(cls=cls) + self.FN(cls=cls)
            if denominator == 0:
                return None
            return self.TN(cls=cls) / denominator
        return np.array([self.NPV(cls=c) for c in range(self.nclasses)], dtype=float)


    # Dependent metrics
    def accuracy(self, cls=None):
        if cls is not None:
            return (
                (self.TP(cls=cls) + self.TN(cls=cls)) /
                (self.TP(cls=cls) + self.TN(cls=cls) + self.FP(cls=cls) + self.FN(cls=cls))
            )
        return np.array([self.accuracy(cls=c) for c in range(self.nclasses)], dtype=float)


    def f_beta(self, beta, cls=None):
        if cls is not None:
            return (
                (1+pow(beta, 2))*
                (
                    (self.precision(cls=cls)*self.recall(cls=cls))/
                    (pow(beta, 2)*self.precision(cls=cls)+self.recall(cls=cls))
                )
            )
        return np.array([self.f_beta(beta, cls=c) for c in range(self.nclasses)], dtype=float)
    

    def FDR(self, cls=None):
        return 1 - self.precision(cls=cls)
    
    def FNR(self, cls=None):
        return 1 - self.recall(cls=cls)

    def FOR(self, cls=None):
        return 1 - self.NPV(cls=cls)
    
    def FPR(self, cls=None):
        return 1 - self.specificity(cls=cls)
    
    # Printing
    def __repr__(self) -> str:
        table = PrettyTable(header=False, border=True, hrules=ALL)
        header = [""] + [f"{name} predicted" for name in self.classnames]
        table.add_row(header)
        for idx, name in enumerate(self.classnames):
            row = [f"{name} true"] + list(self.matrix[idx])
            table.add_row(row)
        return "\n"+table.get_string()
    
    # Plotting
    def plot(self):
        table_green = np.diag(self.TP())
        table_red = self.matrix
        cmap_green = plt.get_cmap("Greens")
        cmap_red = plt.get_cmap("Reds")
        ax = sns.heatmap(
            data=table_green,
            vmin=0, 
            annot=True, 
            cmap=cmap_green,
            cbar=False,
            linewidth=1, 
            )
        sns.heatmap(
            data=table_red,
            vmin=0,
            annot=True,
            cmap=cmap_red,
            cbar=False,
            mask=table_green.astype(bool),
            linewidth=1,
            ax=ax
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_ticklabels(labels=self.classnames)
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_ticklabels(labels=self.classnames, rotation="horizontal")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.show()