'''
this class is gonna be the actual layer in a mlp
'''

import numpy as np
from core.Module import Module
from core.Param import Param

class Linear(Module):

    #linear is a layer with and input dim and an out dim
    # it is a calculation of an activation y = Wx + B


    def __init__(self, input_dim : int, output_dim : int, name: str, bias : bool = True):
        
        #input 
        self.name = name
        self.x_temp = None

        self.input_dim = input_dim
        self.output_dim = output_dim

        #Param setups
        self.W = Param(np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim), "weights")
        self.b = Param(np.zeros(output_dim), "biases") if bias else None


    def forward(self, x):
        #calculate the output given input x
        self.x_temp = x
        y = x @ self.W.value

        if (self.b is not None):
            y = y + self.b.value
    
        return y

    def backward(self, gradient_out):
        #computing the gradient given an input gradient (chain)
        #ipad note for vis

        #theres three things that need a gradient calculated
        # - weight
        # - bias
        # - previous layer activation
        
        self.W.grad += self.x_temp.T @ gradient_out

        if self.b is not None:
            self.b.grad += gradient_out.sum(axis=0)

        gradient_in = gradient_out @ self.W.value.T

        return gradient_in 

    
    def parameters(self):
        #return the params for the grad to be reset
        return [self.W] + ([self.b] if self.b is not None else [])

