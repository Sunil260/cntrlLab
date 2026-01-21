'''
This holde the actication layers ReLu and Sigmoid for now 
'''

from core.Module import Module
import numpy as np
class ReLU(Module):
      
    def __init__(self, name: str):
        self.mask = None
        self.name = name

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask
    #no params so use modules parent classes
    
# class sigmoid(Module):

#     def __init__(self):
#         self.function = 1/(1+np.e**(-x))