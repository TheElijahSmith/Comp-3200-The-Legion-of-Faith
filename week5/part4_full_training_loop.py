import numpy as np

# Trial dataset (Korr's Three Pillars)
tells = np.array([
    [1, 0, 1],  # Sensing 0: foot shift, no guard drop, exhale
    [0, 1, 1],  # Sensing 1: no foot shift, guard drop, exhale
    [0, 0, 1],  # Sensing 2: only exhale
    [1, 1, 1]   # Sensing 3: all three (bluff)
])

# Target column vector (shape (4, 1))
strike = np.array([[1, 1, 0, 0]]).T


def relu(x):
    return np.maximum(0, x)


def relu2deriv(y):
    return (y > 0).astype(float)


# 1. Training loop function
def train(tells, strike, alpha, epochs, hidden_size, seed):
    np.random.seed(seed)
    
    # Weight initialization
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    error_history = []

    for epoch in range(epochs):
        total_error = 0.0
        for i in range(len(tells)):
            layer_0 = tells[i:i+1]   # shape (1, 3)
            target = strike[i:i+1]   # shape (1, 1)

            # Forward pass
            layer_1 = relu(np.dot(layer_0, weights_0_1))  # shape (1, hidden_size)
            layer_2 = np.dot(layer_1, weights_1_2)         # shape (1, 1)

            # Error calculation
            layer_2_error = np.sum((layer_2 - target) ** 2)  # scalar float
            total_error += layer_2_error

            # Backpropagation (with shape comments required by PR rules)
            layer_2_delta = layer_2 - target  # shape (1, 1)
            layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)  # shape (1, hidden_size)

            # SGD Weight updates
            weights_1_2 -= alpha * np.dot(layer_1.T, layer_2_delta)  # shape (hidden_size, 1)
            weights_0_1 -= alpha * np.dot(layer_0.T, layer_1_delta)  # shape (3, hidden_size)

        error_history.append(total_error)

    return weights_0_1, weights_1_2, error_history


if __name__ == "__main__":
    # 2. Main Run (alpha=0.2, epochs=60, hidden_size=4, seed=1)
    alpha = 0.2
    epochs = 60
    hidden_size = 4
    seed = 1

    w01, w12, error_history = train(tells, strike, alpha, epochs, hidden_size, seed)

    print("--- TOTAL ERROR EVERY 10 EPOCHS ---")
    for ep in range(0, epochs, 10):
        print(f"Epoch {ep:2d}: {error_history[ep]:.6f}")
    print(f"Epoch 59: {error_history[-1]:.6f}\n")

    # 3. Final Predictions vs Goals
    print("--- PREDICTIONS VS GOALS ---")
    for i in range(len(tells)):
        l0 = tells[i:i+1]
        l1 = relu(np.dot(l0, w01))
        pred = np.dot(l1, w12)[0, 0]
        goal = strike[i, 0]
        print(f"Sensing {i}: Pred = {pred:.4f} | Goal = {goal}")

    # 4. Learned weights printout
    print("\n--- LEARNED WEIGHTS ---")
    print("weights_0_1.round(2):\n", w01.round(2))
    print("weights_1_2.round(2):\n", w12.round(2))

    '''
    ----------------------------------------------------------------------------------------------
    PART 4: HIDDEN LAYER INTERPRETATION (seed=1, hidden_size=4)
    Unit 0: Dead unit
    Unit 1: Detects foot shift but not guard drop
    Unit 2: Dead unit
    Unit 3: Detects guard drop but not foot shift
    '''

    # 5. Hidden-size Sweep (Minimal code)
    print("\n--- HIDDEN-SIZE SWEEP ---")
    for hs in [1, 2, 4, 8, 16]:
        errors = [f"{train(tells, strike, alpha, epochs, hs, s)[2][-1]:.6f}" for s in [1, 2, 3]]
        print(f"Size {hs}: Seed 1 = {errors[0]}, Seed 2 = {errors[1]}, Seed 3 = {errors[2]}")

    '''
    ----------------------------------------------------------------------------------------------
    PART 4: SWEEP CONCLUSION
    Sizes 1 and 2 fail every time, size 4 is a coin flip, and sizes 8 and 16 succeed reliably.
    The smallest size that works on seed 1 is 4, but the smallest that works reliably is 8 
    because smaller networks often suffer from dead ReLU units depending on weight initialization.
    '''