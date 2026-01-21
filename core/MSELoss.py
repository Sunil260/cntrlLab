'''
error calculation from predicted vs acctual 
mean square error
'''

import numpy as np
from core.Loss import Loss

class MSELoss(Loss):

    def __init__(self):

        self.predicted = None 
        self.actual = None

    def forward(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual
        diff = predicted - actual
        return float(np.mean(diff * diff))

    def backward(self):
        assert self.predicted is not None and self.actual is not None
        diff = self.predicted - self.actual
        grad = (2.0 / diff.size) * diff
        return grad
