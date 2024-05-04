import numpy as np
from confusion import MultiLabelConfusionMatrix


# TODO: Create good tests, multilabel is currently untested.
class TestMultilabel:
    
    y_true = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1]
        ])
    
    y_pred = np.array([
        [1, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1]
        ])
    
    def test_stub(self):
        mlm = MultiLabelConfusionMatrix(self.y_true, self.y_pred)
        print(mlm.binary_matrices)

    def test_properties(self):
        mlm = MultiLabelConfusionMatrix(self.y_true, self.y_pred)
        print("Testing properties")
        print(mlm.TP(cls=0))
        assert mlm.TP(cls=0) == 5
        assert mlm.TN(cls=1) == 2
