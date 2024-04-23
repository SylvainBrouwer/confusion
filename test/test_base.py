import numpy as np
from pytest import approx
from confusion import ConfusionMatrix

import matplotlib.pyplot as plt
import seaborn as sns



class TestSimpleConfusion:

    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])


    def test_precision(self):
        conf = ConfusionMatrix(self.y_true, self.y_pred)
        assert conf.precision() == 1


    def test_recall(self):
        conf = ConfusionMatrix(self.y_true, self.y_pred)
        assert conf.recall() == 0.5


    def test_f1(self):
        conf = ConfusionMatrix(self.y_true, self.y_pred)
        assert conf.f_1() == approx(2*(0.5/1.5), 0.0001)


    def test_aliases(self):
        conf = ConfusionMatrix(self.y_true, self.y_pred)
        assert conf.precision() == conf.PPV()


    def test_print(self):
        conf = ConfusionMatrix(self.y_true, self.y_pred, classnames=("cat", "dog"))
        print(conf)


    def test_plot(self):
        conf = ConfusionMatrix(self.y_true, self.y_pred, classnames=("cat", "dog"))
        conf.plot()