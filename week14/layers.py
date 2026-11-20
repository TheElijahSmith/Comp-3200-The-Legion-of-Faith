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


### Beginning of Part 4 ###

class LSTMCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()
        self.n_inputs, self.n_hidden = n_inputs, n_hidden
        self.n_output = n_output
        self.xf = Linear(n_inputs, n_hidden) # input -> gate (with bias)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)
        self.hf = Linear(n_hidden, n_hidden, bias=False) # hidden -> gate (no bias)
        self.hi = Linear(n_hidden, n_hidden, bias=False)
        self.ho = Linear(n_hidden, n_hidden, bias=False)
        self.hc = Linear(n_hidden, n_hidden, bias=False)
        self.w_ho = Linear(n_hidden, n_output, bias=False) # cell view -> logits
        for layer in (self.xf, self.xi, self.xo, self.xc, self.hf, self.hi, self.ho, self.hc, self.w_ho):
            self.parameters += layer.get_parameters() # in this order

    def forward(self, input, hidden):
        prev_hidden, prev_cell = hidden[0], hidden[1]
        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()
        c = (f * prev_cell) + (i * g) # keep f of the old, write i of the new
        h = o * c.tanh() # expose a gated view of the memory
        return self.w_ho.forward(h), (h, c)

    def init_hidden(self, batch_size = 1):
        h = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        c = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        h.data[:, 0] += 1; c.data[:, 0] += 1 # Trask's tiny symmetry break
        return (h, c)

### Test Block as presented by the build along ##

import numpy as np
from tensor import Tensor
from layers import Embedding, LSTMCell
9
# (a) forward shapes: logits (batch, vocab) and a (h, c) pair
np.random.seed(0)
bs, dim, hidden, vocab = 4, 16, 16, 8
embed = Embedding(vocab, dim)
cell = LSTMCell(dim, hidden, vocab)
hstate = cell.init_hidden(batch_size=bs)
ids = Tensor(np.array([0, 1, 2, 3]))
output, (new_h, new_c) = cell.forward(embed.forward(ids), hstate)
print("logits shape :", output.data.shape)
print("new_h shape :", new_h.data.shape, " new_c shape:", new_c.data.shape)
print("cell parameter tensors:", len(cell.get_parameters()))

# (b) untrained, output-zeroed perplexity at full size (62 characters)
from layers import CrossEntropyLoss
np.random.seed(0)
vocab_size = 62
embed2 = Embedding(vocab_size, 512)
model2 = LSTMCell(512, 512, vocab_size)
model2.w_ho.weight.data *= 0 # zero the output projection
chars = Tensor(np.arange(16)) # any 16 character ids will do
hidden2 = model2.init_hidden(batch_size=16)
output, _ = model2.forward(embed2.forward(chars), hidden2)
loss = CrossEntropyLoss().forward(output, Tensor(np.arange(16)))
print("untrained perplexity:", repr(float(np.exp(loss.data))))
print("== vocab size:", abs(float(np.exp(loss.data)) - vocab_size) < 1e-9)