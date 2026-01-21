'''
general blueprint for loss functions

'''

class Loss:
    def forward(self, predicted, actual):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
