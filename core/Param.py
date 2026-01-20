'''
This is the class for any changeable paramater and the grad that it can be changed by
'''
import numpy as np

class Param:

    def __init__(self, value, name):
        self.value = value
        self.grad = np.zeros_like(value)
        self.param_name = name

    def __str__(self):
        return (f"{self.param_name} \n Value = {self.value} \n Grad = {self.grad}")
    

        

# weight = Param([1], [0.1])
# print(weight)
