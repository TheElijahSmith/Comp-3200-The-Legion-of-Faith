import numpy as np
import time
from data import load_mnist, one_hot
from part3_dropout import evaluate, relu, relu_d, train_dropout

def train_batched(X_train, Y_train, X_val, Y_val, iterations=350):
    np.random.seed(1)
    alpha = 0.1
    hidden_size = 40
    batch_size = 100

    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    train_hist = []
    val_hist = []

    for iteration in range(iterations):
        correct = 0

        for i in range(0, len(X_train), batch_size):
            layer_0 = X_train[i : i + batch_size]
            target = Y_train[i : i + batch_size]

            # forward propogation
            layer_1 = relu(layer_0.dot(W_0_1))

            # make a new mask for the whole batch
            mask = np.random.randint(2, size=layer_1.shape)

            # inverted dropout
            layer_1 = layer_1 * mask * 2

            # Calculate output layer
            layer_2 = layer_1.dot(W_1_2)

            # Backpropagation
            delta_2 = (layer_2 - target) / len(layer_0)
            delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1)

            delta_1 = delta_1 * mask

            # find weight gradients
            gradient_1_2 = layer_1.T.dot(delta_2)
            gradient_0_1 = layer_0.T.dot(delta_1)

            # update the weights
            W_1_2 -= alpha * gradient_1_2
            W_0_1 -= alpha * gradient_0_1

            predictions = np.argmax(layer_2, axis=1)
            actual = np.argmax(target, axis=1)

            correct += np.sum(predictions == actual)

        # get the accuracy after the full training pass
        train_acc = correct / len(X_train)
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)

        train_hist.append(train_acc)
        val_hist.append(val_acc)

    return W_0_1, W_1_2, train_hist, val_hist



if __name__ == "__main__":
    (X_train_full, y_train_full), _ = load_mnist()

    # separate training and validation sets
    X_val = X_train_full[1000:2000]
    Y_val = one_hot(y_train_full[1000:2000])

    X_train = X_train_full[:1000]
    Y_train = one_hot(y_train_full[:1000])

    # Time a pass without batching
    start = time.perf_counter()

    train_dropout(X_train, Y_train, X_val, Y_val, iterations=1)

    single_time = time.perf_counter() - start

    # Time a pass with batching
    start = time.perf_counter()

    train_batched(X_train, Y_train, X_val, Y_val, iterations=1)

    batch_time = time.perf_counter() - start

    # find the speed-up
    speedup = single_time / batch_time

    print(f"Single-image training: {single_time:.4f} seconds")
    print(f"Mini-batch training: {batch_time:.4f} seconds")
    print(f"Speed-up factor: {speedup:.2f}x")

    # Train without batching for 350 iterations
    W1_single, W2_single, tr_single, val_single = train_dropout(
        X_train, Y_train, X_val, Y_val, iterations=350, dropout=True
    )

    # Train with batching for 350 iterations
    W1_batch, W2_batch, tr_batch, val_batch = train_batched(
        X_train, Y_train, X_val, Y_val, iterations=350
    )

    # compare the validation accuracies
    single_val = val_single[-1]
    batch_val = val_batch[-1]

    difference = abs(single_val - batch_val)

    print(f"\nSingle-image validation accuracy: {single_val * 100:.2f}%")
    print(f"Mini-batch validation accuracy: {batch_val * 100:.2f}%")