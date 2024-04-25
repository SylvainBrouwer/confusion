import numpy as np
from confusion import ConfusionMatrix


class TestPlotting:

    def test_three_dims(self):
        dim = 3
        N = 100
        x = np.eye(dim)
        y_true = x[np.random.choice(dim, N)]
        y_pred = x[np.random.choice(dim, N)]
        conf = ConfusionMatrix(y_true, y_pred)
        print(conf)
        conf.plot()


    def test_five_dims_with_classnames(self):
        dim = 5
        N = 100
        x = np.eye(dim)
        classnames = ["banana", "apple", "lime", "mango", "lychee"]
        y_true = x[np.random.choice(dim, N)]
        y_pred = x[np.random.choice(dim, N)]
        conf = ConfusionMatrix(y_true, y_pred, classnames=classnames)
        print(conf)
        conf.plot()
