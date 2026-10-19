import numpy as np
from forms import (
    tanh, tanh_deriv,
    relu, relu_deriv,
    sigmoid, sigmoid_deriv,
    softmax
)
from part3_upgrade import train_step
from data import load_mnist, one_hot

def test_softmax():
    # test with random vals between -25 and 25
    rand_input = (np.random.random((8, 10)) - .5) * 50
    s_maxxed = softmax(rand_input)
    for row in s_maxxed:
        assert abs(np.sum(row) - 1) < 1e-9, "did not sum to 1"
    big_s_max = softmax(np.array([[1000.0, 1001.0]]))
    for row in big_s_max:
        for num in row:
            assert np.isfinite(num), "Not finite with big numbers"
    assert np.isclose(big_s_max[0][0], 0.269, atol=.0001), "Not close to the correct vals"
    assert np.isclose(big_s_max[0][1], 0.731, atol =.0001), "Not close to the correct vals"

def test_ranges():
    x = np.linspace(-15, 15, 1001)
    sig = sigmoid(x)
    tanh_x = tanh(x)
    relu_x = relu(x)
    pos_x = [i for i in x if i>0]
    pos_relu_x = [i for i in relu_x if i>0]
    assert np.min(sig) >= 0, "sigmoid min too low"
    assert np.max(sig) <= 1, "sigmoid max too big"
    assert np.min(tanh_x) >= -1, "tanh_x min too low"
    assert np.max(tanh_x) <= 1, "tanh_x max too big"
    assert np.min(relu_x) >= 0, "relu min too low"
    assert pos_x == pos_relu_x, "relu is transforming positive nums"

def test_relu():
    x = np.linspace(-15, 15, 1001)
    relu_x = relu(x)
    for i in range(len(relu_x)):
        if relu_x[i] == 0:
            assert x[i] <= 0, "relu got a negative sign incorrect"
    assert np.array_equal(relu_deriv(relu(x)), x > 0), "relu deriv has wrong sign"

def test_derivs():
    x = np.random.random(15) * 10
    h = 10**-5
    tanh_analytic_deriv = (tanh(x + h) - tanh(x-h)) / (2*h)
    sig_analytic_deriv = (sigmoid(x + h) - sigmoid(x-h)) / (2*h)
    tanh_x = tanh(x)
    sig_x = sigmoid(x)
    for i in range(len(x)):
        assert abs(tanh_deriv(tanh_x)[i] - tanh_analytic_deriv[i]) <= 1e-4, "tanh deriv does not agree with analytic deriv"
        assert abs(sigmoid_deriv(sig_x)[i] - sig_analytic_deriv[i]) <= 1e-4, "sigmoid deriv does not agree with analytic deriv"


def test_backprop():
    rng = np.random.RandomState(0)
    layer_0 = rng.rand(4, 6) # 4 images, 6 "pixels"
    W_0_1, W_1_2 = rng.randn(6, 5), rng.randn(5, 3)
    Y = np.eye(3)[[0, 2, 1, 2]] # one-hot, 3 classes
    mask = rng.randint(2, size=(4, 5))
    z1 = layer_0 @ W_0_1
    layer_2 = softmax((np.tanh(z1) * mask * 2) @ W_1_2)
    delta_2 = (layer_2 - Y) / 4 # batch average: 4 images
    expected = (delta_2 @ W_1_2.T) * (1 - np.tanh(z1)**2) * mask * 2
    got = train_step(layer_0, Y, W_0_1.copy(), W_1_2.copy(), mask, alpha=0.02)
    # either backward-mask convention: mask * 2 (exact) or mask (Trask)
    assert np.allclose(got, expected) or np.allclose(got, expected / 2)

def test_lin_collapse():
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 3)
    W2 = np.random.randn(3, 2)

    sequencial_l2 = (X @ W1) @ W2
    squashed_l2 = X @ (W1 @ W2)
    relu_l2 = relu(X @ W1) @ W2
    assert np.allclose(sequencial_l2, squashed_l2), "math error"
    assert not np.allclose(relu_l2, squashed_l2), "relu not activating properly"

def test_right_form():
        # load dataset
        (X_train, y_train), (X_test, y_test) = load_mnist()

        # split training and validation data
        X_val = X_train[1000:2000]
        Y_val = one_hot(y_train[1000:2000])

        X_train = X_train[:1000]
        Y_train = one_hot(y_train[:1000])

        alpha = 0.02
        epochs = 15
        hidden_size = 100
        batch_size = 100

        # forms to test
        forms = [
            ("tanh", tanh, tanh_deriv),
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
        tanh_acc = results["tanh"]
        sig_acc = results["sigmoid"]
        assert tanh_acc > sig_acc, "sig acc is greater than tanh when it should not be"







if __name__ == '__main__':
    tests = [name for name in dir() if name.startswith('test_')]
    passed = []
    failed = []
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            passed.append(test_name)
            print(f' PASS: {test_name}')
        except AssertionError as e:
            failed.append(test_name)
            print(f' FAIL: {test_name} -- {e}')
    for name in passed:
        print(f'PASS: {name}')
    for name in failed:
        print(f'FAIL: {name}')
  

"""
In part 4, with the same overall network structures and data, with only a change in hidden layer form, 
we got 3 distinct values for the three different forms. This is because the different forms each have 
unique capabilities in their own rights. Tanh compresses outputs between -1 and 1 allowing for the model 
to still see negative correlations between values and greater slopes around 0, sigmoid compresses outputs 
between 0 and 1, and relu just 0s any negative output allowing a slope always = 1 or 0. Tanh is strong 
and balanced with the capability of negative values so the model can see negative numbers while relu is 
aggressive and allows the model to train quickly. Sigmoid is weaker but still can provide the model a 
different perspective of the datapoint, albeit a slower training time. The Verach, just like a naïve 
neural network creator, have never given their model the chance to view the situation with a different 
perspective that could provide very valuable insights for the model. Without considering other forms, 
you are potentially denying yourself the best view of the data where your model picks 
out the best combination of factors for the situation.
"""