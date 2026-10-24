import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# This function will calculate its derivative without recalculating sigmoid
def sigmoid_deriv(out):
  return out * (1 - out)

def tanh(x):
  return np.tanh(x)

# This will let us calculate its derivative without recalculating tanh
def tanh_deriv(out):
  return 1 - out**2

def relu(x):
  return (x > 0) * x

# This function is ReLU's output tells us whether the input was positive, so we don't need the original input
def relu_deriv(out):
  return (out > 0)

def softmax(z):
  z = z - np.max(z, axis=1, keepdims=True)
  exp_z = np.exp(z)
  return exp_z / np.sum(exp_z, axis=1, keepdims=True)



import matplotlib.pyplot as plt

if __name__ == '__main__':
  # Create input values from -6 to 6
  x = np.linspace(-6, 6, 200)

  # calculate each activation function
  sigmoid_y = sigmoid(x)
  tanh_y = tanh(x)
  relu_y = relu(x)

  # This will draw all three functions on a single graph
  plt.plot(x, sigmoid_y, label="Sigmoid (0, 1)")
  plt.plot(x, tanh_y, label="Tanh (-1, 1)")
  plt.plot(x, relu_y, label="ReLU [0, infinity)")

  # Addings labels
  plt.xlabel("Input (x)")
  plt.ylabel("Output")
  plt.title("Activation Functions")
  plt.legend()
  plt.grid(True)

  # code to save the graph
  plt.savefig("forms_plot.png")
  plt.show()

