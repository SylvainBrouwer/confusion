import numpy as np
import numpy.testing as nptest
import pytest
from confusion import MultiClassConfusionMatrix

class TestMultiClass:
    
    y_true = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ])
    
    y_pred = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1]
        ])


    def test_props(self):
        conf = MultiClassConfusionMatrix(self.y_true, self.y_pred, classnames=["cat", "dog", "bird"])
        nptest.assert_array_equal(conf.TP(), np.array([2, 1, 2]))
        nptest.assert_array_equal(conf.TN(), np.array([4, 4, 5]))
        nptest.assert_array_equal(conf.FP(), np.array([1, 1, 1]))
        nptest.assert_array_equal(conf.FN(), np.array([1, 2, 0]))
    

    def test_metrics(self):
        conf = MultiClassConfusionMatrix(self.y_true, self.y_pred, classnames=["cat", "dog", "bird"])
        assert conf.precision(cls=0) == pytest.approx(2/3, 0.001)
        assert conf.recall(cls=2) == 1.0
        assert conf.specificity(cls=1) == pytest.approx(4/5, 0.001)
        assert conf.NPV(cls=1) == pytest.approx(4/6, 0.001)
        print(conf.precision())
        print(conf.recall())
        print(conf.f_beta(1))
        
