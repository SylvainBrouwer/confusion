class BaseConfusionMatrix:

    def __init__(self):
        pass

    #Properties
    def TP(self):
        pass

    def FP(self):
        pass

    def TN(self):
        pass

    def FN(self):
        pass

    # Metrics
    def precision(self):
        pass

    def recall(self):
        pass

    def specificity(self):
        pass

    def NPV(self):
        pass

    def f_beta(self, beta):
        pass

    def accuracy(self):
        pass

    def FDR(self):
        return 1 - self.precision()
    
    def FNR(self):
        return 1 - self.recall()

    def FOR(self):
        return 1 - self.NPV()
    
    def FPR(self):
        return 1 - self.specificity()

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

