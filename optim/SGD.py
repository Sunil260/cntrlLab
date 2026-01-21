'''
code for stochastic gradient descent

this will shift the params for the gradients that are calculated
This is an optimizer so it should take in the params and a learning rate and then apply it in the step and as it iterates through a minima should be found 
'''

class SGD:

    def __init__(self, params, lr: float = 1e-12):
        self.params = list(params)
        self.lr = float(lr)
        
        
    def step(self)->None:
        for param in self.params:
            param.value -= self.lr * param.grad

