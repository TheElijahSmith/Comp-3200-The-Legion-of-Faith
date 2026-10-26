import numpy as np

from conv import get_patches
from forms import softmax
from part3_train import train_step
from data import load_mnist, one_hot


def test_patch_shape():
    images = np.zeros((100, 28, 28))

    flat_patches, es = get_patches(images)

    assert flat_patches.shape == (67600, 9)


def test_patch_content():
    image = np.arange(784).reshape(1, 28, 28)

    flat_patches, es = get_patches(image)

    expected = np.array([0, 1, 2, 28, 29, 30, 56, 57, 58])

    assert np.array_equal(flat_patches[0], expected)


def test_forward_shape():
    images = np.zeros((3, 28, 28))
    kernels = np.random.random((9, 16))

    flat_patches, es = get_patches(images)

    kernel_out = flat_patches.dot(kernels)
    layer_1 = np.tanh(kernel_out.reshape(3, 10816))

    assert layer_1.shape == (3, 10816)
    assert np.all(layer_1 > -1)
    assert np.all(layer_1 < 1)


def test_training_step():
    rng = np.random.RandomState(0)

    layer_0 = rng.rand(3, 28, 28)
    kernels = 0.5 * rng.randn(9, 16)
    W_1_2 = 0.05 * rng.randn(10816, 10)
    Y = np.eye(10)[[3, 7, 1]]
    mask = rng.randint(2, size=(3, 10816))

    fp, es = get_patches(layer_0)
    ko = fp.dot(kernels)
    l1 = np.tanh(ko.reshape(3, 10816))

    layer_2 = softmax((l1 * mask * 2).dot(W_1_2))

    delta_2 = (layer_2 - Y) / 3
    d1 = delta_2.dot(W_1_2.T) * (1 - l1**2) * mask * 2

    expected = fp.T.dot(d1.reshape(ko.shape))

    got = train_step(layer_0, Y, kernels.copy(),
                     W_1_2.copy(), mask, alpha=0.02)

    assert got.shape == (9, 16)
    assert np.allclose(got, expected) or np.allclose(got, expected / 2)



def test_it_learns():
    np.random.seed(1)

    (images, labels), (test_images, test_labels) = load_mnist()
    train_images = images[:1000].reshape(-1, 28, 28)
    Y = one_hot(labels[:1000])

    val_images = images[1000:2000].reshape(-1, 28, 28)
    val_Y = one_hot(labels[1000:2000])

    kernels = 0.02 * np.random.random((9, 16)) - 0.01
    W_1_2 = 0.2 * np.random.random((10816, 10)) - 0.1

    alpha = 0.02
    batch_size = 128

    for epoch in range(30):
        for s in range(0, len(train_images), batch_size):
            layer_0 = train_images[s:s+batch_size]
            Y_batch = Y[s:s+batch_size]

            mask = np.random.randint(2, size=(len(layer_0), 10816))

            train_step(layer_0, Y_batch, kernels, W_1_2, mask, alpha)

    flat_patches, es = get_patches(val_images)
    kernel_out = flat_patches.dot(kernels)
    layer_1 = np.tanh(kernel_out.reshape(len(val_images), 10816))
    layer_2 = softmax(layer_1.dot(W_1_2))

    accuracy = np.mean(
        np.argmax(layer_2, axis=1) == np.argmax(val_Y, axis=1)
    )

    assert accuracy > 0.5



if __name__ == "__main__":
    tests = [
        test_patch_shape,
        test_patch_content,
        test_forward_shape,
        test_training_step,
        test_it_learns
    ]

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")