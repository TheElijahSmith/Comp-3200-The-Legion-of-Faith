import numpy as np
from tensor import Tensor
from layers import Linear, Sequential
from optim import SGD

np.random.seed(0)

data = Tensor([[0,0], [0,1], [1,0], [1,1]], autograd=True)
target = Tensor([[0], [1], [0], [1]], autograd=True)

model = Sequential([Linear(2, 3), Linear(3, 1)])
optim = SGD(parameters=model.get_parameters(), alpha=0.01)


# Train the model for 10 iterations

for i in range(10):
    pred = model.forward(data)

    loss = ((pred - target) * (pred - target)).sum(0)

    loss.backward()
    optim.step()

    print("Sequential iteration", i, "Loss:", loss.data)




np.random.seed(0)

# two weighted tensors
W1 = Tensor(np.random.randn(2, 3), autograd=True)
W2 = Tensor(np.random.randn(3, 1), autograd=True)



# Create an optimizer for the four trainable tensors
optim = SGD(parameters=[W1, W2], alpha=0.1)


# Train the MLP for 10 iterations

for i in range(10):

    # Forward pass: input to hidden layer
    hidden = data.mm(W1) 

    # Forward pass: hidden layer to output
    pred = hidden.mm(W2)

    # calculate the squared error
    loss = ((pred - target) * (pred - target)).sum(0)

    # backpropagation
    loss.backward()

    # update both weight tensors
    optim.step()

    # print the loss
    print("Bare MLP iteration", i, "Loss:", loss.data)
