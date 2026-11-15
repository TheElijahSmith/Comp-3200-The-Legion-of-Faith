from tensor import Tensor


class SGD:
    def __init__(self, parameters, alpha):
        # Stores a list of tensors
        self.parameters = parameters
        # Learning rate for the function
        self.alpha = alpha


    def step(self, zero=True):
        # Visits all parameters and will update the weights on the gradient. Basically new_weight = old_weight - gradient * learning_rate
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha


            # Clears out gradients post update to prevent them from affecting the next iteration
            if zero:
                p.grad.data *= 0




# test function designed to make sure the above code works.
if __name__ == "__main__":


    from tensor import Tensor


    p = Tensor(10.0, autograd=True) # example tensor
    p.grad = Tensor(2.0) # assumed gradient of 2


    optimizer = SGD([p], alpha=0.1) # optimizer with an alpha of .1


    print("Before:", p.data) # expect 10


    optimizer.step()


    print("After:", p.data) # expect 9.8
