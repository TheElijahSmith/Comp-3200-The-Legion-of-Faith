import numpy as np
from tensor import Tensor


class Layer:
    def __init__(self):
        # Stores all trainable Tensor parameters belonging to this layer
        self.parameters = []

    def get_parameters(self):
        # Returns the list of trainable parameters for the optimizer
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        # Store whether this layer uses a bias term
        self.use_bias = bias

        # Initialize the weight matrix of shape (n_inputs, n_outputs)
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True)
        self.parameters.append(self.weight)

        # Only create and register a bias parameter when bias=True.
        # When bias=False, no bias tensor is created or updated.
        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)
            self.parameters.append(self.bias)

    def forward(self, input):
        # Multiply input by weights, and add broadcasted bias only if bias=True
        if self.use_bias:
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)


class Sequential(Layer):
    def __init__(self, layers=[]):
        super().__init__()
        # Ordered list of layers to run one after another
        self.layers = layers

    def forward(self, input):
        # Pass the input through every layer in sequence
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        # Gather trainable parameters from all layers in the sequence
        params = []
        for l in self.layers:
            params += l.get_parameters()
        return params


class Tanh(Layer):
    def forward(self, input):
        # Apply element-wise tanh activation
        return input.tanh()


class Sigmoid(Layer):
    def forward(self, input):
        # Apply element-wise sigmoid activation
        return input.sigmoid()


class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # Create the learnable lookup table of shape (vocab_size, dim)
        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)
        self.parameters.append(self.weight)

    def forward(self, input):
        # Select the rows matching the integer indices in `input`.
        # No backward code is needed here because Tensor.index_select handles gradients.
        return self.weight.index_select(input)


class CrossEntropyLoss:
    # Not a Layer subclass because it owns no trainable parameters
    def forward(self, input, target):
        # Delegate softmax + negative log-likelihood loss to Tensor.cross_entropy
        return input.cross_entropy(target)