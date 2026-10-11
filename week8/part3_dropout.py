import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist, one_hot


# Activation: turns negative values to zero
def relu(x):
    return (x > 0) * x


# ReLU derivative: 1 if positive, 0 if zero/negative
def relu_d(x):
    return x > 0


# Measure accuracy using the full network (no dropout)
def evaluate(X, Y, W_0_1, W_1_2):
    layer_1 = relu(X.dot(W_0_1))
    layer_2 = layer_1.dot(W_1_2)
    predictions = np.argmax(layer_2, axis=1)
    actual = np.argmax(Y, axis=1)
    return np.mean(predictions == actual)


# Single forward and backward step for one sample
def dropout_step(layer_0, target, W_0_1, W_1_2, mask, alpha):
    l0 = layer_0.reshape(1, -1) if layer_0.ndim == 1 else layer_0
    t = target.reshape(1, -1) if target.ndim == 1 else target

    # Forward pass
    layer_1 = relu(l0.dot(W_0_1))

    if mask is not None:
        mask = np.array(mask).reshape(layer_1.shape)
        # Inverted dropout: turn off half the neurons and double the active ones
        layer_1 = layer_1 * mask * 2

    layer_2 = layer_1.dot(W_1_2)

    # Backward pass
    delta_2 = layer_2 - t
    delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1)

    if mask is not None:
        # A silenced neuron contributed nothing forward, so it gets no gradient backward
        delta_1 = delta_1 * mask

    # Update weights
    W_1_2 -= alpha * layer_1.T.dot(delta_2)
    W_0_1 -= alpha * l0.T.dot(delta_1)

    return layer_2


# Training loop with optional dropout
def train_dropout(X_train, Y_train, X_val, Y_val, iterations=350, dropout=True):
    np.random.seed(1)
    alpha, hidden_size = 0.005, 40

    # Initialize weights in range [-0.1, 0.1)
    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    train_hist = []
    val_hist = []

    for it in range(iterations):
        correct = 0
        for i in range(len(X_train)):
            l0 = X_train[i : i + 1]
            t = Y_train[i : i + 1]

            if dropout:
                # 50% chance mask for hidden layer
                mask = np.random.randint(2, size=(1, hidden_size))
            else:
                mask = None

            layer_2 = dropout_step(l0, t, W_0_1, W_1_2, mask, alpha)

            if np.argmax(layer_2) == np.argmax(t):
                correct += 1

        train_acc = correct / len(X_train)
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)

        train_hist.append(train_acc)
        val_hist.append(val_acc)

    return W_0_1, W_1_2, train_hist, val_hist


if __name__ == "__main__":
    # Load and split dataset
    (X_train_full, y_train_full), _ = load_mnist()
    X_val, Y_val = X_train_full[1000:2000], one_hot(y_train_full[1000:2000])
    X_train, Y_train = X_train_full[:1000], one_hot(y_train_full[:1000])

    # Run without dropout
    W1_no, W2_no, tr_no, val_no = train_dropout(
        X_train, Y_train, X_val, Y_val, iterations=350, dropout=False
    )
    eval_tr_no = evaluate(X_train, Y_train, W1_no, W2_no)

    # Run with dropout
    W1_do, W2_do, tr_do, val_do = train_dropout(
        X_train, Y_train, X_val, Y_val, iterations=350, dropout=True
    )
    eval_tr_do = evaluate(X_train, Y_train, W1_do, W2_do)

    # Plot curves and save plot
    plt.figure(figsize=(8, 5))
    plt.plot(tr_do, label="Training Accuracy (Running, Mask ON)")
    plt.plot(val_do, label="Validation Accuracy (Mask OFF)")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Part 3: Accuracy Curves with Dropout")
    plt.legend()
    plt.grid(True)
    plt.savefig("curves_part3.png")
    plt.close()

    # Print summary metrics
    best_pass_no = int(np.argmax(val_no))
    best_pass_do = int(np.argmax(val_do))

    print("=" * 70)
    print("SUMMARY COMPARISON: WITH vs WITHOUT DROPOUT")
    print("=" * 70)
    print(f"{'Metric':<42} | {'No Dropout':<10} | {'With Dropout':<10}")
    print("-" * 70)
    print(f"{'Final Train Acc (Running Count)':<42} | {tr_no[-1]*100:6.1f}%   | {tr_do[-1]*100:6.1f}%")
    print(f"{'Final Train Acc evaluate() (Mask OFF)':<42} | {eval_tr_no*100:6.1f}%   | {eval_tr_do*100:6.1f}%")
    print(f"{'Best Val Acc (Pass)':<42} | {val_no[best_pass_no]*100:5.1f}% ({best_pass_no}) | {val_do[best_pass_do]*100:5.1f}% ({best_pass_do})")
    print(f"{'Final Val Acc':<42} | {val_no[-1]*100:6.1f}%   | {val_do[-1]*100:6.1f}%")
    print("=" * 70)