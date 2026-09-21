import numpy as np

# Dataset setup (Korr's Three Pillars)
# Input matrix shape: (4 sensings, 3 binary features each)
# Features per column: [foot shift, guard drop, exhale]
tells = np.array([
    [1, 0, 1],  # Sensing 0: foot shift, no guard drop, exhale
    [0, 1, 1],  # Sensing 1: no foot shift, guard drop, exhale
    [0, 0, 1],  # Sensing 2: only exhale
    [1, 1, 1]   # Sensing 3: all three (bluff)
])

# Target column vector: 1 = strike imminent, 0 = hold
# Formatted as a column matrix with shape (4, 1) for dot product math
strike = np.array([[1, 1, 0, 0]]).T


# Activation functions
def relu(x):
    # Elementwise Rectified Linear Unit (ReLU): converts negative values to 0
    return np.maximum(0, x)


def relu2deriv(y):
    # Derivative of ReLU: returns 1.0 for positive outputs y > 0, otherwise 0.0
    # Used during backprop to block gradients through inactive/dead neurons
    return (y > 0).astype(float)


# Training loop function using Stochastic Gradient Descent (SGD)
def train(tells, strike, alpha, epochs, hidden_size, seed):
    # Seed the NumPy random number generator for reproducible weight initialization
    np.random.seed(seed)
    
    # Randomly initialize weight matrices with values scaled in the range [-1.0, 1.0]
    # weights_0_1 connects inputs (3) to hidden neurons (hidden_size)
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    # weights_1_2 connects hidden neurons (hidden_size) to output neuron (1)
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    # Track total squared error at the end of each epoch
    error_history = []

    # Outer loop: iterate over total number of training epochs
    for epoch in range(epochs):
        total_error = 0.0
        
        # Inner loop: Stochastic Gradient Descent (SGD) — process sample by sample
        for i in range(len(tells)):
            # Extract single sample row while preserving 2D shape (1, 3)
            layer_0 = tells[i:i+1]   # shape (1, 3)
            # Extract single target value while preserving 2D shape (1, 1)
            target = strike[i:i+1]   # shape (1, 1)

            # Forward Pass
            # Compute hidden layer activations using ReLU applied to weighted sum
            layer_1 = relu(np.dot(layer_0, weights_0_1))  # shape (1, hidden_size)
            # Compute final output prediction (linear sum with no activation)
            layer_2 = np.dot(layer_1, weights_1_2)         # shape (1, 1)

            # Error Calculation
            # Calculate squared error for current sample: (prediction - target)^2
            layer_2_error = np.sum((layer_2 - target) ** 2)  # scalar float
            # Accumulate error to compute total error for full epoch
            total_error += layer_2_error

            # Backpropagation
            # Derivative of loss with respect to prediction: (pred - target)
            layer_2_delta = layer_2 - target  # shape (1, 1)
            
            # Backpropagate error through transposed weights_1_2 to hidden layer,
            # then multiply by relu2deriv to mask out inactive hidden units
            layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)  # shape (1, hidden_size)

            # SGD Weight Updates
            # Update weights_1_2 using outer product of layer_1 activation and layer_2_delta
            weights_1_2 -= alpha * np.dot(layer_1.T, layer_2_delta)  # shape (hidden_size, 1)
            # Update weights_0_1 using outer product of layer_0 input and layer_1_delta
            weights_0_1 -= alpha * np.dot(layer_0.T, layer_1_delta)  # shape (3, hidden_size)

        # Append total error recorded across all 4 samples for this epoch
        error_history.append(total_error)

    return weights_0_1, weights_1_2, error_history


if __name__ == "__main__":
    # Main run setup and training
    # Hyperparameters required by prompt
    alpha = 0.2
    epochs = 60
    hidden_size = 4
    seed = 1

    # Train model and get final weights and error history list
    w01, w12, error_history = train(tells, strike, alpha, epochs, hidden_size, seed)

    # Print total error every 10 epochs to observe convergence from ~1 to < 10^-3
    print("--- TOTAL ERROR EVERY 10 EPOCHS ---")
    for ep in range(0, epochs, 10):
        print(f"Epoch {ep:2d}: {error_history[ep]:.6f}")
    # Print error at final epoch 59
    print(f"Epoch 59: {error_history[-1]:.6f}\n")

    # Final predictions vs goals
    print("--- PREDICTIONS VS GOALS ---")
    # Evaluate model predictions for each sensing sample using learned weights
    for i in range(len(tells)):
        l0 = tells[i:i+1]                             # Input sample, shape (1, 3)
        l1 = relu(np.dot(l0, w01))                    # Hidden layer forward activation
        pred = np.dot(l1, w12)[0, 0]                  # Final prediction scalar
        goal = strike[i, 0]                           # Target goal scalar
        print(f"Sensing {i}: Pred = {pred:.4f} | Goal = {goal}")

    # Learned weights printout
    print("\n--- LEARNED WEIGHTS ---")
    # Display rounded weight matrices
    print("weights_0_1.round(2):\n", w01.round(2))
    print("weights_1_2.round(2):\n", w12.round(2))

    '''
    PART 4: HIDDEN LAYER INTERPRETATION (seed=1, hidden_size=4)
    Unit 0: Dead unit
    Unit 1: Detects foot shift but not guard drop
    Unit 2: Dead unit
    Unit 3: Detects guard drop but not foot shift
    '''

    # Hidden-size sweep execution
    print("\n--- HIDDEN-SIZE SWEEP ---")
    # Test sizes {1, 2, 4, 8, 16} across seeds {1, 2, 3}
    for hs in [1, 2, 4, 8, 16]:
        # For each size, train across 3 seeds and extract final epoch total error (index [2][-1])
        errors = [f"{train(tells, strike, alpha, epochs, hs, s)[2][-1]:.6f}" for s in [1, 2, 3]]
        print(f"Size {hs}: Seed 1 = {errors[0]}, Seed 2 = {errors[1]}, Seed 3 = {errors[2]}")

    '''
    PART 4: SWEEP CONCLUSION
    Sizes 1 and 2 fail every time, size 4 is a coin flip, and sizes 8 and 16 succeed reliably.
    The smallest size that works on seed 1 is 4, but the smallest that works reliably is 8 
    because smaller networks often suffer from dead ReLU units depending on weight initialization.
    '''
