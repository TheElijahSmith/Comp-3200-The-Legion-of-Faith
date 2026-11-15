import numpy as np
from tensor import Tensor
from optim import SGD
from layers import Linear, Sequential, Tanh, Sigmoid, MSELoss


def test_multi_use_accumulation():
    # Part 1 acceptance test: a leaf tensor used twice must sum incoming gradients
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)
    d = a + b
    e = b + c
    f = d + e
    f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
    assert np.array_equal(b.grad.data, np.array([2, 2, 2, 2, 2])), (
        f"Expected [2, 2, 2, 2, 2], got {b.grad.data}"
    )


def test_add_passes_gradient_through():
    # g = (a + b) + (c + d) should pass ones back to every leaf tensor
    a = Tensor([1.0, 2.0, 3.0], autograd=True)
    b = Tensor([4.0, 5.0, 6.0], autograd=True)
    c = Tensor([7.0, 8.0, 9.0], autograd=True)
    d = Tensor([10.0, 11.0, 12.0], autograd=True)
    g = (a + b) + (c + d)
    g.backward(Tensor(np.ones_like(g.data)))
    ones = np.ones_like(a.data)
    for leaf in (a, b, c, d):
        assert np.array_equal(leaf.grad.data, ones), (
            f"Expected {ones}, got {leaf.grad.data}"
        )


def test_numeric_gradient_check_mul_sum():
    # Numeric gradient check on y = (x * x).sum(0) testing mul and sum backward branches
    eps = 1e-5
    x_vals = np.array([2.0, 3.0])
    x = Tensor(x_vals.copy(), autograd=True)
    y = (x * x).sum(0)
    y.backward()

    analytic = x.grad.data.copy()
    assert np.allclose(analytic, np.array([4.0, 6.0]), atol=1e-6), (
        f"Expected [4.0, 6.0], got {analytic}"
    )

    numeric = np.zeros_like(x_vals)
    for i in range(len(x_vals)):
        x_plus = x_vals.copy()
        x_plus[i] += eps
        f_plus = float((Tensor(x_plus) * Tensor(x_plus)).sum(0).data)

        x_minus = x_vals.copy()
        x_minus[i] -= eps
        f_minus = float((Tensor(x_minus) * Tensor(x_minus)).sum(0).data)

        numeric[i] = (f_plus - f_minus) / (2 * eps)

    max_diff = float(np.max(np.abs(analytic - numeric)))
    print(f"    [Mul/Sum Grad Check] analytic={analytic}, numeric={numeric}, max_diff={max_diff:.2e}")
    assert max_diff < 1e-4, f"Max diff {max_diff} >= 1e-4"


def test_every_step_decreases_loss():
    # Part 4 Sequential model (seed=0, alpha=0.01): loss must strictly decrease over 10 steps
    np.random.seed(0)
    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = Tensor([[0], [1], [0], [1]], autograd=True)
    model = Sequential([Linear(2, 3), Linear(3, 1)])
    optim = SGD(parameters=model.get_parameters(), alpha=0.01)

    losses = []
    for _ in range(10):
        pred = model.forward(data)
        loss = ((pred - target) * (pred - target)).sum(0)
        loss.backward()
        optim.step()
        losses.append(float(loss.data.item()))

    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1], (
            f"Loss failed to decrease at step {i}: {losses[i]} >= {losses[i - 1]}"
        )


def test_shape_and_parameter_count():
    # Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()]) must return 4 parameter Tensors
    model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
    params = model.get_parameters()
    assert len(params) == 4, f"Expected 4 parameter tensors, got {len(params)}"

    expected_shapes = [(2, 3), (3,), (3, 1), (1,)]
    actual_shapes = [p.data.shape for p in params]
    assert actual_shapes == expected_shapes, (
        f"Expected shapes {expected_shapes}, got {actual_shapes}"
    )


def test_the_wait():
    # Non-leaf tensor s = x + y used twice: engine must wait until all children report
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)
    x = Tensor([1, 2, 3, 4, 5], autograd=True)
    y = Tensor([1, 1, 1, 1, 1], autograd=True)
    s = x + y
    loss = (s + a) + (s + c)
    loss.backward(Tensor(np.array([1, 1, 1, 1, 1])))
    assert np.array_equal(x.grad.data, np.array([2, 2, 2, 2, 2])), (
        f"Expected [2, 2, 2, 2, 2], got {x.grad.data}"
    )


def test_reuse_through_mm():
    # Non-leaf tensor h = X.mm(W) used twice in matrix multiplications
    X = Tensor([[1., 2.], [3., 4.]], autograd=True)
    W = Tensor([[0.5, -0.5], [1., 0.25]], autograd=True)
    A = Tensor([[1., 0.], [2., 1.]], autograd=True)
    B = Tensor([[0., 1.], [1., 3.]], autograd=True)
    h = X.mm(W)
    (h.mm(A) + h.mm(B)).sum(0).sum(0).backward()
    expected = [[8.0, 28.0], [12.0, 42.0]]
    assert np.allclose(W.grad.data, expected), (
        f"Expected {expected}, got {W.grad.data.tolist()}"
    )


def test_whole_network_gradient_check():
    # Numeric gradient check across the entire Part 5 nonlinear network
    # Runs through mm, add, expand, tanh, sigmoid, sub, mul, and sum at once
    np.random.seed(0)
    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = Tensor([[0], [1], [0], [1]], autograd=True)
    model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
    criterion = MSELoss()

    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()

    # Nudge entry [0, 0] of the first Linear layer's weight matrix by +-eps
    eps = 1e-5
    w = model.layers[0].weight
    analytic_val = float(w.grad.data[0, 0])
    orig_val = float(w.data[0, 0])

    w.data[0, 0] = orig_val + eps
    loss_plus = float(criterion.forward(model.forward(data), target).data.item())

    w.data[0, 0] = orig_val - eps
    loss_minus = float(criterion.forward(model.forward(data), target).data.item())

    # Restore original weight
    w.data[0, 0] = orig_val

    numeric_val = (loss_plus - loss_minus) / (2 * eps)
    abs_diff = abs(analytic_val - numeric_val)
    print(f"    [Full Net Grad Check W1[0,0]] analytic={analytic_val:.8f}, numeric={numeric_val:.8f}, abs_diff={abs_diff:.2e}")
    assert abs_diff < 1e-4, f"Network gradient check failed: abs_diff {abs_diff} >= 1e-4"


if __name__ == "__main__":
    tests = [
        test_multi_use_accumulation,
        test_add_passes_gradient_through,
        test_numeric_gradient_check_mul_sum,
        test_every_step_decreases_loss,
        test_shape_and_parameter_count,
        test_the_wait,
        test_reuse_through_mm,
        test_whole_network_gradient_check,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__} -> {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")