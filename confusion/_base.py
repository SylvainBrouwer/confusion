from abc import ABC, abstractmethod

class BaseConfusionMatrix(ABC):

    #Properties
    @property
    @abstractmethod
    def TP(self):
        pass

    @property
    @abstractmethod
    def FP(self):
        pass

    @property
    @abstractmethod
    def TN(self):
        pass

    @property
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

