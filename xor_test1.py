import numpy as np

from core.Sequential import Sequential
from nn.Linear import Linear
from nn.Activations import ReLU
from core.MSELoss import MSELoss
from optim.SGD import SGD

np.random.seed(1)

# XOR dataset
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=float)
Y = np.array([[0.],[1.],[1.],[0.]], dtype=float)

#2 input -> 4 neuron -> 1 output

xor_model = Sequential(
    Linear(2,4,"first hidden layer"),
    ReLU("1st relu"),
    Linear(4,1, "output layer")
)

loss_function = MSELoss()

optimizer = SGD(xor_model.parameters(), lr = 0.01)

for step in range(1000):
    #every training ste reset the gradients
    xor_model.zero_grad()

    #get the new prediction with the last updates
    y_pred = xor_model.forward(X)
    #calculate the loss compared to the know outputs from the xor
    loss = loss_function.forward(y_pred, Y)

    # get the grad of the loss wrt the y_predicted
    grad = loss_function.backward()
    #back prop the gradient throught the params
    xor_model.backward(grad)

    #apply the gradients with the lr 
    optimizer.step()

    if step % 50 == 0:
        print(step, loss)
        print(f"y predicted = {y_pred}")

# Evaluate
pred = xor_model.forward(X)
print("predictions:\n", pred)
print("rounded:\n", (pred > 0.5).astype(int))

