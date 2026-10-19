
import numpy as np
import matplotlib.pyplot as plt

from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax


def train_step(layer_0, Y_batch, W_0_1, W_1_2, mask,
               alpha, hidden_f=tanh, hidden_d=tanh_deriv):
    # forward prop
    layer_1 = hidden_f(layer_0 @ W_0_1)

    # derivative before dropout
    slope = hidden_d(layer_1)

    # apply dropout
    layer_1 *= mask * 2

    # find output probabilities
    layer_2 = softmax(layer_1 @ W_1_2)

    # find batch size
    batch_size = layer_0.shape[0]

    # find output layer error
    delta_2 = (layer_2 - Y_batch) / batch_size

    # find hidden layer error
    delta_1 = delta_2 @ W_1_2.T * slope

    # apply dropout mask during backprop
    # using the exact gradient convention (mask * 2)
    delta_1 *= mask * 2

    # update weight matrices
    W_1_2 -= alpha * layer_1.T @ delta_2
    W_0_1 -= alpha * layer_0.T @ delta_1

    return delta_1


def main():
    # load dataset
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # split training and validation data
    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])

    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])

    Y_test = one_hot(y_test)


    np.random.seed(1)

    alpha = 0.02
    epochs = 300
    hidden_size = 100
    batch_size = 100

    W_0_1 = 0.02 * np.random.random((784, hidden_size)) - 0.01
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1


    val_accuracies = []

    # training time
    for epoch in range(epochs):

        # process the training images in batches
        for s in range(0, len(X_train), batch_size):

            # current batch
            layer_0 = X_train[s:s + batch_size]
            Y_batch = Y_train[s:s + batch_size]

            # new dropout mask
            mask = np.random.randint(2, size=(len(layer_0), hidden_size))

            # train on current batch
            train_step(layer_0, Y_batch, W_0_1, W_1_2, mask, alpha)

        # validation accuracy without dropout
        val_hidden = tanh(X_val @ W_0_1)
        val_scores = val_hidden @ W_1_2

        # find predicted and actual digits
        val_predictions = np.argmax(val_scores, axis=1)
        val_actual = np.argmax(Y_val, axis=1)

        # get accuracy
        val_accuracy = np.mean(val_predictions == val_actual) * 100

        # store accuracy for plotting
        val_accuracies.append(val_accuracy)

        # show the progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.2f}%")

    # final test evaluation without dropout
    test_hidden = tanh(X_test @ W_0_1)
    test_scores = test_hidden @ W_1_2

    # get preds and actual digits
    test_predictions = np.argmax(test_scores, axis=1)
    test_actual = np.argmax(Y_test, axis=1)

    # final test accuracy
    test_accuracy = np.mean(test_predictions == test_actual) * 100

    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # plot 

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), val_accuracies)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("MNIST Validation Accuracy - Tanh and Softmax")

    plt.grid(True)
    plt.savefig("week9/upgrade_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
