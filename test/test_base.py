import numpy as np
from confusion import ConfusionMatrix


def test_precision():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    conf = ConfusionMatrix(y_true, y_pred)
    assert conf.precision() == 1


def test_recall():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    conf = ConfusionMatrix(y_true, y_pred)
    assert conf.recall() == 0.5


def test_aliases():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    conf = ConfusionMatrix(y_true, y_pred)
    assert conf.precision() == conf.PPV()
