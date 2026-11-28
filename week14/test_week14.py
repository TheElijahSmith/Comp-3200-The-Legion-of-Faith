import numpy as np

from tensor import Tensor
from layers import Embedding, LSTMCell, CrossEntropyLoss
from optim import SGD
from exploding_gradients import exploding, lstm_cell_state_horizon

def test_index_select_backward():

    matrix = Tensor(np.eye(5), autograd=True)

    indices = Tensor(np.array([[1, 2, 3], [2, 3, 4]]))

    selected = matrix.index_select(indices)

    selected.backward(Tensor(np.ones_like(selected.data)))

    expected = np.zeros((5, 5))

    for index in indices.data.flatten():
        expected[index] += 1

    assert np.array_equal(matrix.grad.data, expected)

    print("Index select backward works!")


def test_untrained_perplexity():

    np.random.seed(0)
    vocab_size = 62
    embedding = Embedding(vocab_size, 512)
    model = LSTMCell(512, 512, vocab_size)

    model.w_ho.weight.data *= 0
    chars = Tensor(np.array([0]))

    hidden = model.init_hidden(batch_size=1)

    embedded = embedding.forward(chars)
    output, _ = model.forward(embedded, hidden)

    loss = CrossEntropyLoss().forward(
        output,
        Tensor(np.array([0]))

    )
    perplexity = np.exp(float(loss.data))

    assert np.isclose(perplexity, vocab_size)

    print("Untrained perplexity:", perplexity)


def test_one_step_decreases_loss():

    np.random.seed(0)
    vocab_size = 5
    embed_size = 4
    hidden_size = 4
    batch_size = 2

    embedding = Embedding(vocab_size, embed_size)

    model = LSTMCell(
        embed_size,
        hidden_size,
        vocab_size
    )

    loss_function = CrossEntropyLoss()

    parameters = (
        embedding.get_parameters()
        + model.get_parameters()
    )

    optimizer = SGD(parameters, alpha=0.01)

    inputs = Tensor(np.array([0, 1]))
    targets = Tensor(np.array([1, 2]))


    def calculate_loss():
        hidden = model.init_hidden(batch_size)
        embedded = embedding.forward(inputs)
        prediction, hidden = model.forward(embedded, hidden)
        return loss_function.forward(prediction, targets)

    before_loss = float(calculate_loss().data)

    loss = calculate_loss()
    loss.backward()
    optimizer.step()
    after_loss = float(calculate_loss().data)

    assert after_loss < before_loss

    print("Loss before", before_loss)
    print("Loss after:", after_loss)


def test_exploding_gradients():

    actual = exploding()

    expected = [5 ** step for step in range(1, 11)]
    assert np.array_equal(actual, expected)

  gated = lstm_cell_state_horizon(forget_gate=True)
  ungated = lstm_cell_state_horizon(forget_gate=False)

    assert max(gated) < 2.0
    assert ungated[-1] > 4.0
    assert ungated[-1] > max(gated) * 3

    print("exploding gradients:", actual)
    print("Gated final state:", gated[-1])
    print("Ungated final state:", ungated[-1])


def run_test(test):

    try:
        test()
        print("Pass:", test.__name__)
        return True

    except Exception as error:
        print("Fail:", test.__name__)
        print("Reason:", error)
        return False


if __name__ == "__main__":

    tests = [
        test_index_select_backward,
        test_untrained_perplexity,
        test_one_step_decreases_loss,
        test_exploding_gradients
    ]

    results = []

    for test in tests:
        results.append(run_test(test))
        print()

    passed = sum(results)

    print("Tests passed:", passed, "/", len(tests))
