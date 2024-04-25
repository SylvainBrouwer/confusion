from abc import ABC, abstractmethod
import numpy as np


def _convert_to_bool_np(input):
    if isinstance(input, (list, np.ndarray)):
        return np.array(input, dtype=bool)

class BaseConfusionMatrix(ABC):
    """
    Abstract base class for confusion matrices. Specifies methods to be implemented and provides several dependent methods and method aliases.
    """

    @staticmethod
    def _get_classnames(classnames, nclasses):
        if classnames is None:
            return np.array(range(nclasses))
        return np.array(classnames)


    #Properties
    @abstractmethod
    def TP(self):
        """
        Abstract method for true positives.
        """
        pass

    @abstractmethod
    def FP(self):
        """
        Abstract method for false positives.
        """
        pass

    @abstractmethod
    def TN(self):
        """
        Abstract method for true negatives.
        """
        pass

    @abstractmethod
    def FN(self):
        """
        Abstract method for false negatives.
        """
        pass

    # Base metrics
    @abstractmethod
    def precision(self):
        """
        Abstract method for precision score.
        """
        pass

    @abstractmethod
    def recall(self):
        """
        Abstract method for recall score.
        """
        pass

    @abstractmethod
    def specificity(self):
        """
        Abstract method for specificity.
        """
        pass

    @abstractmethod
    def NPV(self):
        """
        Abstract method for negative predictive value.
        """
        pass

    # Dependent metrics
    @abstractmethod
    def f_beta(self, beta):
        """
        Abstract method for F-score.
        """
        pass

    @abstractmethod
    def accuracy(self):
        """
        Abstract method for accuracy.
        """
        pass

    def FDR(self, cls=None):
        return 1 - self.precision(cls=cls)
    
    def FNR(self, cls=None):
        return 1 - self.recall(cls=cls)

    def FOR(self, cls=None):
        return 1 - self.NPV(cls=cls)
    
    def FPR(self, cls=None):
        return 1 - self.specificity(cls=cls)

    # Metric aliases
    def f_1(self):
        """
        Method alias for f_beta(1).
        """
        return self.f_beta(1)

    def PPV(self):
        """
        Method alias for precision().
        """
        return self.precision()
    
    def sensitivity(self):
        """
        Method alias for recall().
        """
        return self.recall()
    
    def TPR(self):
        """
        Method alias for recall().
        """
        return self.recall()
    
    def TNR(self):
        """
        Method alias for specificity().
        """
        return self.specificity()

