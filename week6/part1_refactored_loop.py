import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def train(tells, strike, alpha, epochs, hidden_size, seed):
    """
    Trains a 2-layer neural network with ReLU hidden activation using SGD.
    Organized into four visual movements with shape annotations on every math line.
    """
    np.random.seed(seed)

    # Input and output dimensions based on input shapes
    input_size = tells.shape[1] if tells.ndim > 1 else len(tells[0])
    
    # Ensure goal array is 2D with shape (N, 1) or (N, output_dim)
    goals = np.array(strike).reshape(-1, 1)
    output_size = goals.shape[1]

    # Initialize weight matrices with uniform distribution in [-1, 1]
    weights_0_1 = 2 * np.random.random((input_size, hidden_size)) - 1  # (3, 4)
    weights_1_2 = 2 * np.random.random((hidden_size, output_size)) - 1  # (4, 1)

    error_history = []

    for epoch in range(epochs):
        total_error = 0.0

        for i in range(len(tells)):
            # Extract single observation with batch size of 1
            layer_0 = tells[i : i + 1]  # (1, 3)
            goal = goals[i : i + 1]  # (1, 1)

            # FORWARD
            layer_1 = relu(layer_0 @ weights_0_1)  # (1, 4)
            layer_2 = layer_1 @ weights_1_2  # (1, 1)

            # COMPARE
            error = np.sum((layer_2 - goal) ** 2) 
            total_error += error 

            # BACKWARD
            layer_2_delta = layer_2 - goal  # (1, 1)
            layer_1_delta = (layer_2_delta @ weights_1_2.T) * relu_deriv(layer_1)  # (1, 4)

            # UPDATE
            weights_1_2 -= alpha * (layer_1.T @ layer_2_delta)  # (4, 1)
            weights_0_1 -= alpha * (layer_0.T @ layer_1_delta)  # (3, 4)

        error_history.append(total_error)

    return error_history


if __name__ == "__main__":
    # Example Trial Dataset setup (matching Week 5)
    tells = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])
    strike = np.array([1, 1, 0, 0])

    # Run refactored loop
    history = train(tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1)
    print(f"Final epoch error: {history[-1]:.10f}")