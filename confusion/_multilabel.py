import numpy as np
from ._base import BaseConfusionMatrix, _convert_to_bool_np
from ._confusionmatrix import ConfusionMatrix

class MultiLabelConfusionMatrix(BaseConfusionMatrix):

    def __init__(self, y_true, y_pred, classnames=None):
        y_true = _convert_to_bool_np(y_true)
        y_pred = _convert_to_bool_np(y_pred)
        self.nclasses = y_true.shape[1]
        self.classnames = self._get_classnames(classnames, self.nclasses)

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape.")
        if self.classnames.shape[0] != self.nclasses:
            raise ValueError(f"Number of provided class names does not match number of classes: {self.classnames.shape[0]} != {self.nclasses}")
        
        self.binary_matrices = []
        for cls in range(self.nclasses):
            labels_true = y_true[:, cls]
            labels_pred = y_pred[:, cls]
            cls_true = np.stack((1-labels_true, labels_true)).transpose()
            cls_pred = np.stack((1-labels_pred, labels_pred)).transpose()
            cls_matrix = ConfusionMatrix(cls_true, cls_pred, classnames=[cls, "other"])
            self.binary_matrices.append(cls_matrix)
    

    # Helper
    def _get_metric(self, method_name, cls=None):
        if cls is not None:
            return getattr(self.binary_matrices[cls], method_name)(cls=0)
        return np.array([getattr(self.binary_matrices[cls], method_name)(cls) for cls in range(self.nclasses)])

    # Properties
    @property
    def binary_matrixes(self):
        return self.binary_matrices
    
    def TP(self, cls=None):
        return self._get_metric("TP", cls)

    def TN(self, cls=None):
        return self._get_metric("TN", cls)

    def FP(self, cls=None):
        return self._get_metric("FP", cls)

    def FN(self, cls=None):
        return self._get_metric("FN", cls)

    # Base metrics
    def precision(self, cls=None):
        return self._get_metric("precision", cls)
    
    def recall(self, cls=None):
        return self._get_metric("recall", cls)

    def NPV(self, cls=None):
        return self._get_metric("NPV", cls)

    def specificity(self, cls=None):
        return self._get_metric("specificity", cls)

    # Dependent metrics
    # TODO: make helper work with beta parameter
    def f_beta(self, beta, cls=None):
        if cls is not None:
            return self.binary_matrices[cls].f_beta(beta, cls=0)
        return np.array([self.f_beta(beta, cls) for cls in range(self.nclasses)])
    
    def accuracy(self, cls=None):
        return self._get_metric("accuracy", cls)
    