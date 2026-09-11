# week4/part1_multi_input.py
import numpy as np

# No Numpy version

# the three inputs for sensing 0: blade_angle, balance, breath
sensing_0 = [8.5, 0.65, 1.2]
weights = [0.1, 0.2, -0.1]
goal = 1.0
alpha = 0.01
iterations = 10

def w_sum(a, b):
    # multiplies each pair of numbers and adds them all up (dot product)
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def ele_mul(scalar, vector):
    # multiplies one number by every item in a list
    output = []
    for i in range(len(vector)):
        output.append(scalar * vector[i])
    return output

def gradient_descent_multi(input, weights, true, alpha, iterations):
    # keeps its own copy so it doesn't change the list we were handed
    weights = list(weights)

    error_history = []
    weight_history = []

    for iteration in range(iterations):
        # predict: combine the 3 inputs and 3 weights into one number
        pred = w_sum(input, weights)

        # compare: how far off we are, and in which direction
        error = (pred - true) ** 2
        delta = pred - true

        # learn: the same delta gets spread across the 3 weights,
        # scaled by each weight's own input
        weight_deltas = ele_mul(delta, input)
        for i in range(len(weights)):
            weights[i] -= alpha * weight_deltas[i]

        # save a copy, not the live list, or every entry would end up
        # pointing at the same final weights
        error_history.append(error)
        weight_history.append(list(weights))

        print(f"Iteration {iteration}: pred={pred:.5f}  error={error:.6f}  weights={[round(w,5) for w in weights]}")

    return weights, error_history, weight_history

# Numpy Version

def gradient_descent_multi_np(input, weights, true, alpha, iterations):
    # convert to arrays so numpy can do the math elementwise
    input = np.array(input)
    weights = np.array(weights, dtype=float)

    error_history = []
    weight_history = []

    for iteration in range(iterations):
        # np.dot does the same job as our w_sum
        pred = np.dot(input, weights)
        error = (pred - true) ** 2
        delta = pred - true

        # numpy multiplies delta by every input automatically, no loop needed
        weight_deltas = delta * input
        weights = weights - alpha * weight_deltas

        error_history.append(error)
        weight_history.append(weights.copy())

    return weights, error_history, weight_history

# Verify both versions agree

plain_weights, plain_errors, plain_hist = gradient_descent_multi(sensing_0, weights, goal, alpha, iterations)
numpy_weights, numpy_errors, numpy_hist = gradient_descent_multi_np(sensing_0, weights, goal, alpha, iterations)

print("\nFinal weights (plain vs numpy):")
for i in range(len(plain_weights)):
    a = plain_weights[i]
    b = numpy_weights[i]
    # abs(a - b) < 1e-10 checks "close enough" instead of exact equality
    print(f"weight {i}: plain={a:.8f}   numpy={b:.8f}   match={abs(a - b) < 1e-10}")

'''
Which weight changed the most / least, and why:
weight 0 (blade_angle) moves the most, weight 1 (balance) moves the least. 
weight_delta = delta * input, and delta is the same shared number for all three weights since they come from one prediction.
The only thing that differs between weights is their own input value, so the weight attached to the biggest input (blade_angle = 8.5)
 gets the biggest push every iteration, and the one attached to the smallest input (balance = 0.65) gets pushed the least.

'''
