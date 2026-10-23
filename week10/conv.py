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

  return expanded.reshaoe(es[0] = es[1], 9), es
