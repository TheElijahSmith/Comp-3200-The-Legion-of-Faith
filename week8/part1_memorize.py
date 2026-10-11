import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist, one_hot

(X_train, y_train), (X_test, _y_test) = load_mnist()
X_val, Y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])

def relu(x): return (x > 0) * x
def relu_d(x): return (x > 0) # > 0, not >= 0: an off unit gets no blame

def evaluate(X, Y, W_0_1, W_1_2): # Nothing here updates weights #
    # Forward Pass
    layer_1 = relu(X.dot(W_0_1))
    layer_2 = layer_1.dot(W_1_2)

    # Compare Pred with the actual Acc
    predictions = np.argmax(layer_2, axis=1)
    actual = np.argmax(Y, axis=1)

    return np.mean(predictions == actual)

def train(X_train, y_train, X_val, Y_val, iterations = 350):
    np.random.seed(1) # seed FIRST, then initialise
    alpha, hidden_size = 0.005, 40
    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1 # [-0.1, 0.1)
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    # Arrays for plot later on
    train_hist = []
    val_hist = []

    for it in range(iterations):
        correct = 0
        for i in range(1000): # first 1,000 entries only
            # Forward Pass
            layer_0 = X_train[i:i+1]
            layer_1 = relu(layer_0.dot(W_0_1))
            layer_2 = layer_1.dot(W_1_2)

            # Score
            correct += int(np.argmax(layer_2) == np.argmax(y_train[i]))

            # Backward Pass
            delta_2 = layer_2 - y_train[i:i+1]
            delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1)
            W_1_2 -= alpha * layer_1.T.dot(delta_2)
            W_0_1 -= alpha * layer_0.T.dot(delta_1)

        # Accuracy and Validation measured during pass
        train_acc = correct / 1000
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)

        train_hist.append(train_acc)
        val_hist.append(val_acc)

        print(
            f"iter {it:3d} "
            f"train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )

    return W_0_1, W_1_2, train_hist, val_hist

if __name__ == "__main__":

    # Trains network & collects Acc History
    W_0_1, W_1_2, train_hist, val_hist = train(X_train, y_train, X_val, Y_val)

    plt.figure(figsize = (8, 5))
    plt.plot(train_hist, label="Training Accuracy")
    plt.plot(val_hist, label="Validation Accuracy)")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("curves_part1.png")
    plt.close()

    best_iter = np.argmax(val_hist)

    print(
        f"\nValidation accuracy peaked at iteration "
        f"{best_iter} with accuracy "
        f"{val_hist[best_iter]:.3f}"
    )