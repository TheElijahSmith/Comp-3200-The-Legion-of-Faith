from tensor import Tensor
from layers import Linear, Sequential, Tanh, Sigmoid, MSELoss
from optim import SGD
import numpy as np


np.random.seed(0)

# training inputs and expected outputs
data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
target = Tensor([[0], [1], [0], [1]], autograd=True)

# make the nonlinear neural network
model = Sequential([
    Linear(2, 3),
    Tanh(),
    Linear(3, 1),
    Sigmoid()
])

# loss function and optimizer
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=1)

# train model for 10 iterations
losses = []

for i in range(10):
    # forward pass
    pred = model.forward(data)

    # calculate prediction error
    loss = criterion.forward(pred, target)

    # backpropagate gradients
    loss.backward()

    # update weights and clear gradients
    optim.step()

    # save and display loss
    losses.append(float(np.sum(loss.data)))
    print(f"Iteration {i}: Loss = {loss.data}")