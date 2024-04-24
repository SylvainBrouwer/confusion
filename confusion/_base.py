from abc import ABC, abstractmethod
import numpy as np


def _convert_to_bool_np(input):
    if isinstance(input, (list, np.ndarray)):
        return np.array(input, dtype=bool)

class BaseConfusionMatrix(ABC):

    #Properties
    @abstractmethod
    def TP(self):
        pass

    @abstractmethod
    def FP(self):
        pass

    @abstractmethod
    def TN(self):
        pass

    @abstractmethod
    def FN(self):
        pass

    # Base metrics
    @abstractmethod
    def precision(self):
        pass

    @abstractmethod
    def recall(self):
        pass

    @abstractmethod
    def specificity(self):
        pass

    @abstractmethod
    def NPV(self):
        pass

    # Dependent metrics
    @abstractmethod
    def f_beta(self, beta):
        pass

    @abstractmethod
    def accuracy(self):
        pass

    def FDR(self):
        pass
    
    def FNR(self):
        pass

    def FOR(self):
        pass
    
    def FPR(self):
        pass

    # Metric aliases
    def f_1(self):
        return self.f_beta(1)

    def PPV(self):
        return self.precision()
    
    def sensitivity(self):
        return self.recall()
    
    def TPR(self):
        return self.recall()
    
    def TNR(self):
        return self.specificity()

