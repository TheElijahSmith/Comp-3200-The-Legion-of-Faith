import numpy as np
import matplotlib.pyplot as plt

from data import load_mnist, one_hot
from forms import (
    tanh, tanh_deriv,
    relu, relu_deriv,
    sigmoid, sigmoid_deriv
)
from part3_upgrade import train_step


def main():
    # load dataset
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # split training and validation data
    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])

    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])

    alpha = 0.02
    epochs = 300
    hidden_size = 100
    batch_size = 100

    # forms to test
    forms = [
        ("tanh", tanh, tanh_deriv),
        ("relu", relu, relu_deriv),
        ("sigmoid", sigmoid, sigmoid_deriv)
    ]

    results = {}

    # train network for each activation form
    for name, hidden_f, hidden_d in forms:
        print(f"Training network with {name}...")

        np.random.seed(1)

        W_0_1 = 0.02 * np.random.random((784, hidden_size)) - 0.01
        W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

        for epoch in range(epochs):

            # process the training images in batches
            for s in range(0, len(X_train), batch_size):
                layer_0 = X_train[s:s + batch_size]
                Y_batch = Y_train[s:s + batch_size]

                # new dropout mask
                mask = np.random.randint(2, size=(len(layer_0), hidden_size))

                # train on current batch
                train_step(layer_0, Y_batch, W_0_1, W_1_2, mask, alpha, hidden_f, hidden_d)

        # validation accuracy without dropout
        val_hidden = hidden_f(X_val @ W_0_1)
        val_scores = val_hidden @ W_1_2

        # find predicted and actual digits
        val_predictions = np.argmax(val_scores, axis=1)
        val_actual = np.argmax(Y_val, axis=1)

        # get accuracy
        val_accuracy = np.mean(val_predictions == val_actual) * 100
        results[name] = val_accuracy

        print(f"Final Validation Accuracy ({name}): {val_accuracy:.2f}%\n")

    # print table
    print("=" * 30)
    print(f"{'Form':<10} | {'Val Accuracy':<15}")
    print("-" * 30)
    for name, acc in results.items():
        print(f"{name:<10} | {acc:.2f}%")
    print("=" * 30)

    # plot comparison
    names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, accuracies)

    plt.xlabel("Hidden Activation Form")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Comparison of Hidden Layer Forms")
    plt.ylim(0, 100)

    plt.grid(True)
    plt.savefig("compare_plot.png")
    plt.show()


if __name__ == "__main__":
    main()