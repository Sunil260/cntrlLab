'''
Collection of the layers to make up a MLP type neural network 
-built of layers like linear, relu, softmax
modularity is shown here
'''

from core.Module import Module

class Sequential(Module):

    def __init__(self, *layers:Module):
        #take the tuple of a list of params added of layers and put it into the objects list of layers
   
        self.layers = list(layers)

    def forward(self, x):
        #pass the input x through the layers
        print(self.layers)
        for layer in self.layers:
            print(type(layer))
            x = layer.forward(x)

        return x
    def backward(self, grad_out):
        #the first grad out comes from the loss and the last layer
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)

        return grad_out
    
    def parameters(self):
        #return all params from all the layers
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
