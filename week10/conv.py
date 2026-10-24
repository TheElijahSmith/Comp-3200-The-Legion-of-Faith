# Part 1
import numpy as np

def get_patches(layer):
  batch_size = layer.shape[0]
  patches = []

  for r in range(layer.shape[1] - 3 + 1):
    for c in range(layer.shape[2] - 3 + 1):
      patch = layer[:, r:r+3, c:c+3]
      patches.append(patch.reshape(batch_size, 1, 3, 3))

  expanded = np.concatenate(patches, axis=1)
  es = expanded.shape

  return expanded.reshape(es[0] * es[1], 9), es

# Part 2
def init_kernels():
    return 0.02 * np.random.random((9, 16)) - 0.01

def conv_forward(layer_0, kernels, W_1_2=None, mask=None):
    # Use actual batch size n to handle smaller last batch
    n = layer_0.shape[0]

    # Extract patches: (n * 676, 9)
    flat_patches, es = get_patches(layer_0)

    # Dot product with 16 kernels: (n * 676, 16)
    kernel_out = flat_patches.dot(kernels)

    # Reshape per image and apply tanh: (n, 10816)
    layer_1_raw = np.tanh(kernel_out.reshape(n, 676 * 16))

    # Save slope before dropout
    slope = 1.0 - layer_1_raw ** 2

    # Apply dropout mask and survivor scaling
    if mask is not None:
        layer_1 = layer_1_raw * mask * 2
    else:
        layer_1 = layer_1_raw

    # Softmax classifier output if W_1_2 is provided
    layer_2 = None
    if W_1_2 is not None:
        logits = layer_1.dot(W_1_2)
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        layer_2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return flat_patches, kernel_out, layer_1, slope, layer_2