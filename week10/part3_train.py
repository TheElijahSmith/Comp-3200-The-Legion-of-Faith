import numpy as np
import matplotlib.pyplot as plt

from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax
from conv import get_patches


def train_step(layer_0, Y_batch, kernels, W_1_2, mask, alpha):

    # Get all 3x3 patches
    flat_patches, patch_shape = get_patches(layer_0)

    # Convolution
    kernel_out = flat_patches.dot(kernels)
    # shape = (batch * 676, 16)

    # Flatten into one vector per image
    layer_1 = kernel_out.reshape(layer_0.shape[0], -1)

    # Activation
    layer_1 = tanh(layer_1)

    # Derivative before dropout
    slope = tanh_deriv(layer_1)

    # Dropout
    layer_1 *= mask * 2

    # Output layer
    layer_2 = softmax(layer_1.dot(W_1_2))

    # =====================
    # Backpropagation
    # =====================

    delta_2 = (layer_2 - Y_batch) / layer_0.shape[0]

    delta_1 = delta_2.dot(W_1_2.T) * slope

    delta_1 *= mask * 2

    W_1_2 -= alpha * layer_1.T.dot(delta_2)

    delta_1_reshaped = delta_1.reshape(kernel_out.shape)

    k_update = flat_patches.T.dot(delta_1_reshaped)

    kernels -= alpha * k_update

    return k_update


def main():

    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Reshape into images
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    # Week 10 split
    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])

    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])

    Y_test = one_hot(y_test)

    np.random.seed(1)

    alpha = 2
    epochs = 300
    batch_size = 128

    # 16 kernels
    kernels = 0.02 * np.random.random((9, 16)) - 0.01

    hidden_size = 26 * 26 * 16  # 10816

    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    val_accuracies = []

    # =====================
    # Training
    # =====================

    for epoch in range(epochs):

        for s in range(0, len(X_train), batch_size):

            layer_0 = X_train[s:s + batch_size]
            Y_batch = Y_train[s:s + batch_size]

            mask = np.random.randint(
                2,
                size=(len(layer_0), hidden_size)
            )

            train_step(
                layer_0,
                Y_batch,
                kernels,
                W_1_2,
                mask,
                alpha
            )

        # =====================
        # Validation
        # =====================

        flat_patches, patch_shape = get_patches(X_val)

        kernel_out = flat_patches.dot(kernels)

        val_hidden = tanh(
            kernel_out.reshape(X_val.shape[0], -1)
        )

        val_scores = val_hidden.dot(W_1_2)

        val_predictions = np.argmax(val_scores, axis=1)
        val_actual = np.argmax(Y_val, axis=1)

        val_accuracy = np.mean(
            val_predictions == val_actual
        ) * 100

        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Validation Accuracy: {val_accuracy:.2f}%"
            )

    # =====================
    # Test Set
    # =====================

    flat_patches, patch_shape = get_patches(X_test)

    kernel_out = flat_patches.dot(kernels)

    test_hidden = tanh(
        kernel_out.reshape(X_test.shape[0], -1)
    )

    test_scores = test_hidden.dot(W_1_2)

    test_predictions = np.argmax(test_scores, axis=1)
    test_actual = np.argmax(Y_test, axis=1)

    test_accuracy = np.mean(
        test_predictions == test_actual
    ) * 100

    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), val_accuracies)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Week 10 Convolution Network")

    plt.grid(True)

    plt.savefig("week10_conv_plot.png")
    plt.show()


if __name__ == "__main__":
    main()