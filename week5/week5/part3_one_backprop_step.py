import numpy as np

np.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2deriv(y):
    return (y > 0).astype(int)


def one_step(layer_0, target, weights_0_1, weights_1_2, alpha):
    # Forward propogate
    layer_1 = relu(layer_0.dot(weights_0_1))
    layer_2 = layer_1.dot(weights_1_2)

    # Calculate error
    layer_2_error = np.sum((layer_2 - target) ** 2)

    # Back propogate
    layer_2_delta = layer_2 - target
    layer_1_delta = (
        layer_2_delta.dot(weights_1_2.T)
        * relu2deriv(layer_1)
    )

    # Update weights
    weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
    weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

    # return weights
    return weights_0_1, weights_1_2, layer_2, layer_2_error


tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])

strike = np.array([[1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 4

weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1


layer_0 = tells[0:1]
target = strike[0:1]


layer_1_before = relu(layer_0.dot(weights_0_1))
layer_2_before = layer_1_before.dot(weights_1_2)
layer_2_error_before = np.sum((layer_2_before - target) ** 2)

print("Layer 2 before:", layer_2_before)
print("Layer 2 error before:", layer_2_error_before)
print("Weights 0-1 shape before:", weights_0_1.shape)
print("Weights 1-2 shape before:", weights_1_2.shape)


weights_0_1, weights_1_2, layer_2, layer_2_error = one_step(layer_0, target, weights_0_1, weights_1_2, alpha)


layer_1_after = relu(layer_0.dot(weights_0_1))
layer_2_after = layer_1_after.dot(weights_1_2)
layer_2_error_after = np.sum((layer_2_after - target) ** 2)

print("Layer 2 after:", layer_2_after)
print("Layer 2 error after:", layer_2_error_after)
print("Weights 0-1 shape after:", weights_0_1.shape)
print("Weights 1-2 shape after:", weights_1_2.shape)

# ------------------------------------------------- PART 4 -----------------------------------------------------
# Manual verification for weights_1_2[1, 0]:
# new = old - alpha * layer_1.T.dot(layer_2_delta)
# new = 0.75623487 - 0.2 * (0.51828245 * -0.60805673)
# new = 0.81926390

# ------------------------------------------------- PART 5 -----------------------------------------------------
# We transpose the weights because we're propagating the error backwards so the transpose reverses the order 
# so that the output error can distribute the error to the hidden neurons. We multiply by relu2deriv(layer 1)
# to only use the neurons that were active after relu. Anything that equals 0 doesn't affect anything.