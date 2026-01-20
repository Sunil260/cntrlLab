'''
This is the parent class that provides the structure to the layer classes so that they all have required methods

'''

class Module:

    #functions needed fwd, bwd, params, zero_grad
    def __init__(self):
        print("hi")
        pass
    
    def forward(self,input):
        #if it gets here then its not written for the layer
        raise NotImplementedError
    
    def backward(self, gradient):
       raise NotImplementedError
    
    def parameters(self):
        return [] 

    def zero_grad(self):
        #reset the gradients
        for p in self.parameters():
            p.grad.fill(0)


        

        
