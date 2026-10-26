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

    delta_2 = (layer_2 - Y_batch) / layer_0.shape[0] # batch avg
    delta_1 = delta_2.dot(W_1_2.T) * slope # slope saved before mask
    delta_1 *= mask * 2 # forward was * mask * 2
    W_1_2 -= alpha * layer_1.T.dot(delta_2)

    delta_1_reshaped = delta_1.reshape(kernel_out.shape) # (batch*676, 16)
    k_update = flat_patches.T.dot(delta_1_reshaped) # (9, 16)
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


    #=======================
    # Kernel visualizaiton
    #=======================
    kernel3x3s = kernels.reshape(3, 3, 16)
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10,6))
    for i in range(len(axs.flat)):
        axs.flat[i].imshow(kernel3x3s[:,:,i], cmap = 'gray')
        axs.flat[i].axis("off")
        axs.flat[i].set_title(i+1, fontsize = 9, pad = 4)
    plt.savefig("week10/kernels.png")
    plt.show()

"""
1, 6, and 15 detect a bottom right corner of an object. Plots 2 and 4 and maybe 9 seem to be some sort of diagonal line.
Plot 10 and maybe 14 detect a vertical edge. And 12 is a top left corner of an object. I can't confidently say what
any of the rest of the plots are detecting, as their patterns do not some to be any regular shapes.
"""

if __name__ == "__main__":
    main()