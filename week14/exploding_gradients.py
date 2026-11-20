import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# shows gradients shrinking through a sigmoid RNN
def vanishing():
    W = np.array([[1, 4], [4, 1]])
    activation = sigmoid(np.array([1.0, 0.01]))
    grad = np.ones(2)
    gradients = []
    activations = []

    # forward pass
    for i in range(10):
        activation = sigmoid(activation.dot(W))
        activations.append(activation)

    # backward pass
    for i in reversed(range(10)):
        derivative = activations[i] * (1 - activations[i])
        grad = (grad * derivative).dot(W.T)

        # save largest gradient each iteration
        gradients.append(float(np.max(np.abs(grad))))

    return gradients

# shows gradients growing through a ReLU RNN
def exploding():
    W = np.array([[1, 4], [4, 1]])
    activation = np.maximum(0, np.array([1.0, 0.01]))
    grad = np.ones(2)
    gradients = []
    activations = []

    # forward pass but with ReLU
    for i in range(10):
        activation = np.maximum(0, activation.dot(W))
        activations.append(activation)

    # backward pass
    for i in reversed(range(10)):
        derivative = (activations[i] > 0).astype(float)
        grad = (grad * derivative).dot(W.T)

        # save largest gradient each iteration
        gradients.append(float(np.max(np.abs(grad))))

    return gradients


def lstm_cell_state_horizon(steps=50, forget_gate=True, seed=0):
    rng = np.random.RandomState(seed)
    c, norms = np.zeros(16), []
    for _ in range(steps):
        x = rng.randn(16)
        f_pre = rng.randn(16) # always drawn: same writes either way
        f = sigmoid(f_pre) if forget_gate else np.ones(16)
        i = sigmoid(rng.randn(16))
        c = f * c + i * np.tanh(x) # keep f of the old, write i of the new
        norms.append(float(np.max(np.abs(c))))
    return norms



if __name__ == "__main__":
    gated = lstm_cell_state_horizon(forget_gate=True)
    ungated = lstm_cell_state_horizon(forget_gate=False)

    print(f"Gated final state: {gated[-1]:.4f}")
    print(f"Ungated final state: {ungated[-1]:.4f}")

    plt.plot(range(1, 51), gated, label="With Forget Gate")
    plt.plot(range(1, 51), ungated, label="Without Forget Gate")

    plt.xlabel("Time Steps")
    plt.ylabel("Maximum Absolute Cell State")
    plt.title("Gated vs. Ungated LSTM Cell State")
    plt.legend()
    plt.grid(True)

    plt.savefig("gated_vs_ungated.png")
    plt.show()