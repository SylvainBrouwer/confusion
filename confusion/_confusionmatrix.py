import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable, ALL
from ._base import BaseConfusionMatrix, _convert_to_bool_np


class ConfusionMatrix(BaseConfusionMatrix):
    """
    Class for creating (multiclass) confusion matrices.

    ### Parameters
    `y_true`: array-like of shape (n_samples, n_classes), one hot encoded.
        True one-hot encoded labels.

    `y_pred`: array-like of shape (n_samples, n_classes), one hot encoded.
        Predicted one-hot encoded labels.

    `classnames` [optional]: array-like of shape (n_classes)
        Names for target classes.

    ### Attributes
    `nclasses`: int
        Number of classes.

    `classnames`: np.ndarray of shape (n_classes)
        Numpy array containing names for classes.
    
    `matrix`: np.ndarray of shape (n_classes, n_classes)
        Numpy array containing confusion matrix values.
    """

    def __init__(self, y_true, y_pred, classnames=None):
        y_true = _convert_to_bool_np(y_true)
        y_pred = _convert_to_bool_np(y_pred)
        self.nclasses = y_true.shape[1]
        self.classnames = self._get_classnames(classnames, self.nclasses)

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape.")
        if (y_true.sum(axis=1) != 1).any() or (y_pred.sum(axis=1) != 1).any():
            raise ValueError("ConfusionMatrix does not support multi-label inputs.")
        if self.classnames.shape[0] != self.nclasses:
            raise ValueError(f"Number of provided class names does not match number of classes: {self.classnames.shape[0]} != {self.nclasses}")
        
        self.matrix = np.zeros((self.nclasses, self.nclasses), dtype=int)
        trues = np.where(y_true == 1)[1]
        for true, pred in zip(trues, y_pred):
            self.matrix[true] += pred


    # Properties
    def TP(self, cls=None) -> int | np.ndarray[np.int_]:
        """
        Return true positives

        ### Parameters
        `cls`: class index
            When cls is None, an array containing true positives for all classes is returned.

        ### Returns
        True positives for cls, or array containing true positives for all classes.
        """
        if cls is not None:
            return self.matrix[cls, cls]
        return np.diagonal(self.matrix)
    

    def TN(self, cls=None) -> int | np.ndarray[np.int_]:
        """
        Return true negatives. 

        ### Parameters
        `cls`: class index
            When cls is None, an array containing true negatives for all classes is returned.

        ### Returns
        True negatives for cls, or array containing true negatives for all classes.
        """
        if cls is not None:
            return np.sum(self.matrix) - self.matrix[:, cls].sum() - self.matrix[cls, :].sum() + self.TP(cls=cls)
        return np.array([self.TN(cls=c) for c in range(self.nclasses)])
 

    def FP(self, cls=None) -> int | np.ndarray[np.int_]:
        """
        Return false positives.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing false positives for all classes is returned.

        ### Returns
        False positives for cls, or array containing false positives for all classes.
        """
        if cls is not None:
            return self.matrix[:, cls].sum() - self.TP(cls=cls)
        return np.array([self.FP(cls=c) for c in range(self.nclasses)])    


    def FN(self, cls=None) -> int | np.ndarray[np.int_]:
        """
        Return false negatives.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing false negatives for all classes is returned.

        ### Returns
        False negatives for cls, or array containing false negatives for all classes.
        """
        if cls is not None:
            return self.matrix[cls, :].sum() - self.TP(cls=cls)
        return np.array([self.FN(cls=c) for c in range(self.nclasses)])
    
    
    # Base metrics
    def precision(self, cls=None) -> float | np.ndarray[np.double]:
        """
        Calculate precision.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing precision scores for all classes is returned.

        ### Returns
        Precision score for cls, or array containing precision scores for all classes.
        """
        if cls is not None:
            denominator = self.TP(cls=cls) + self.FP(cls=cls)
            if denominator == 0:
                return None
            return self.TP(cls=cls) / denominator
        return np.array([self.precision(cls=c) for c in range(self.nclasses)], dtype=float)


    def recall(self, cls=None) -> float | np.ndarray[np.double]:
        """
        Calculate recall.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing recall scores for all classes is returned.

        ### Returns
        Recall score for cls, or array containing recall scores for all classes.
        """
        if cls is not None:
            denominator = self.TP(cls=cls) + self.FN(cls=cls)
            if denominator == 0:
                return None
            return self.TP(cls=cls) / denominator
        return np.array([self.recall(cls=c) for c in range(self.nclasses)], dtype=float)


    def specificity(self, cls=None) -> float | np.ndarray[np.double]:
        """
        Calculate specificity.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing specificity scores for all classes is returned.

        ### Returns
        Specificity score for cls, or array containing specificity scores for all classes.
        """
        if cls is not None:
            denominator = self.TN(cls=cls) + self.FP(cls=cls)
            if denominator == 0:
                return None
            return self.TN(cls=cls) / denominator
        return np.array([self.specificity(cls=c) for c in range(self.nclasses)], dtype=float)


    def NPV(self, cls=None) -> float | np.ndarray[np.double]:
        """
        Calculate NPV.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing NPV scores for all classes is returned.

        ### Returns
        NPV score for cls, or array containing NPV scores for all classes.
        """
        if cls is not None:
            denominator = self.TN(cls=cls) + self.FN(cls=cls)
            if denominator == 0:
                return None
            return self.TN(cls=cls) / denominator
        return np.array([self.NPV(cls=c) for c in range(self.nclasses)], dtype=float)


    # Dependent metrics
    def accuracy(self, cls=None) -> float | np.ndarray[np.double]:
        """
        Calculate accuracy.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing accuracy scores for all classes is returned.

        ### Returns
        Accuracy score for cls, or array containing accuracy scores for all classes.
        """
        if cls is not None:
            return (
                (self.TP(cls=cls) + self.TN(cls=cls)) /
                (self.TP(cls=cls) + self.TN(cls=cls) + self.FP(cls=cls) + self.FN(cls=cls))
            )
        return np.array([self.accuracy(cls=c) for c in range(self.nclasses)], dtype=float)


    def f_beta(self, beta, cls=None) -> float | np.ndarray[np.double]:
        """
        Calculate F-score for a given parameter beta.

        ### Parameters
        `cls`: class index
            When cls is None, an array containing F-scores for all classes is returned.

        `beta`: beta parameter for the F-score.

        ### Returns
        F-score for cls, or array containing F-score for all classes.
        """
        if beta < 0:
            raise ValueError("Parameter beta must be larger than 0.")
        if cls is not None:
            return (
                (1+pow(beta, 2))*
                (
                    (self.precision(cls=cls)*self.recall(cls=cls))/
                    (pow(beta, 2)*self.precision(cls=cls)+self.recall(cls=cls))
                )
            )
        return np.array([self.f_beta(beta, cls=c) for c in range(self.nclasses)], dtype=float)

    
    # Printing
    def __repr__(self) -> str:
        """
        Confusion matrix object representation, using PrettyTable.

        ### Returns
        PrettyTable string representation for confusion matrix.
        """
        table = PrettyTable(header=False, border=True, hrules=ALL)
        header = [""] + [f"{name} predicted" for name in self.classnames]
        table.add_row(header)
        for idx, name in enumerate(self.classnames):
            row = [f"{name} true"] + list(self.matrix[idx])
            table.add_row(row)
        return "\n"+table.get_string()
    

    # Plotting
    def plot(self) -> None:
        """
        Plot confusionmatrix, using seaborn.
        """
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
            mask = np.logical_not(table_green.astype(bool)),
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

    # Saving
    def to_csv(self, filename) -> None:
        """
        Save matrix to csv.

        ### Parameters
        `filename`: str
            Path to save to.
        """
        np.savetxt(filename, self.matrix, fmt="%d", delimiter=",")