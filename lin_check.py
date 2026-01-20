''' just check in on. the seqquential model '''
import numpy as np 
from core.Sequential import Sequential
from nn.Linear import Linear
from nn.Activations import ReLu

#make some dummy layers
h_layer_1 = Linear( input_dim = 3, output_dim = 4, name="hidden layer 1", bias = True)

relu_1 = ReLu("relu1")

h_layer_2  = Linear(input_dim= 4 , output_dim= 2, name="hidden layer 2", bias= True)

print("Made the sublayers")
#put them into a sequential 

model = Sequential(h_layer_1, relu_1, h_layer_2)


# Forward shape check
x = np.random.randn(5, 3)      # batch=5, input_dim=3
print(x)
y = model.forward(x)
print("y shape:", y.shape)
assert y.shape == (5, 2)
print("y is\n", y)

# Backward shape check
grad_out = np.random.randn(5, 2)
print("GradOut" , grad_out)
dx = model.backward(grad_out)
print("dx is\n",dx)
# print("dx shape:", dx.shape)
assert dx.shape == (5, 3)

# Parameter + zero_grad check
params = model.parameters()
print("num params:", len(params))
assert len(params) > 0

for p in params:
    print(p)


model.zero_grad()
for p in params:
    assert np.allclose(p.grad, 0)

print("Sequential stack sanity check PASSED")