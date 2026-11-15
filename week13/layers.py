from tensor import Tensor
import numpy as np


# Base Class
class Layer:
    def __init__(self):
        self.parameters = list() # the Tensors SGD must visit


    def get_parameters(self):
        return self.parameters # returns parameters at the layer


# Dense layer
class Linear(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in) # Weight matrix to prevent activations from changing size too much mid training
        self.weight = Tensor(W, autograd=True) # weight matrix to be learned
        self.bias = Tensor(np.zeros(n_out), autograd=True) # Bias vector to be learned. Init as zeros
        self.parameters += [self.weight, self.bias]


    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data)) # matrix multiplication


# Chain multiple layers together
class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers # ordered list of layers


    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input) # feed each into the next
        return input


    def get_parameters(self):
        params = [] # collect all params from every layer
        for l in self.layers:
            params += l.get_parameters() # gather every layer's weights
        return params

# Main function meant to test the above code
if __name__ == "__main__":


    # Build a small network:
    # 2 inputs > 3 hidden neurons > 1 output neuron
    model = Sequential([
        Linear(2, 3),
        Linear(3, 1)
    ])


    # Gather all trainable parameters
    params = model.get_parameters()


    print("Number of parameters:", len(params))
    print("Expected: 4")
    print()


    # Linear(2,3)
    #   weight -> (2,3)
    #   bias   -> (3,)
    #
    # Linear(3,1)
    #   weight -> (3,1)
    #   bias   -> (1,)
    #
    # Total trainable tensors = 4


    for i, p in enumerate(params):
        print(f"Parameter {i}:")
        print(p.data)
        print()


    # Create one sample containing two features
    x = Tensor(np.array([[1.0, 2.0]]), autograd=True)


    # Run the input through the network
    output = model.forward(x)


    print("Network output:")
    print(output.data)
    print()


    print("Output shape:")
    print(output.data.shape)
    print("Expected shape: (1, 1)")
