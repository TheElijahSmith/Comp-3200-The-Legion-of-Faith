
#===================================== PART 1 & PART 1B==============================================

# example tests for Part 1 and Part 1b -- these get merged into the

from part1_multi_input import gradient_descent_multi
from part1b_rescaled import normalize

def test_normalize_max_becomes_one():
    # the biggest value in the channel should land exactly on 1.0
    result = normalize([8.5, 9.5, 9.9, 9.0])
    assert result[2] == 1.0

def test_normalize_does_not_modify_original():
    # normalize should return a NEW list, not change the one it was given
    original = [8.5, 9.5, 9.9, 9.0]
    original_copy = list(original)
    normalize(original)
    assert original == original_copy

def test_gradient_descent_multi_error_goes_down():
    # error on the last iteration should be smaller than on the first
    _, errors, _ = gradient_descent_multi([8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 10)
    assert errors[-1] < errors[0]

def test_gradient_descent_multi_prediction_close_to_goal():
    # after enough iterations on an easy sensing, prediction should be near the goal
    final_weights, _, _ = gradient_descent_multi([2.0, 1.0, 3.0], [0.2, 0.5, -0.1], 1.0, 0.01, 200)
    pred = sum(w * i for w, i in zip(final_weights, [2.0, 1.0, 3.0]))
    assert abs(pred - 1.0) < 0.01

if __name__ == "__main__":
    tests = [
        test_normalize_max_becomes_one,
        test_normalize_does_not_modify_original,
        test_gradient_descent_multi_error_goes_down,
        test_gradient_descent_multi_prediction_close_to_goal,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError:
            print(f"FAIL: {test.__name__}")

#========================================================================================================

#===================================== PART 2 ===========================================================

from part2_multi_output import gradient_descent_outputs

def test_predictions_move_closer_to_targets():
    input = .65
    weights = [.3, .2, .9]
    trues = [0, 1, 0]

    initial_predictions = [input * w for w in weights]

    final_weights, error_history, weight_history = gradient_descent_outputs(input, weights, trues, .1, 10)

    final_predictions = [input * w for w in final_weights]

    for i in range(3):
        initial_error = abs(initial_predictions[i] - trues[i])
        final_error = abs(final_predictions[i] - trues[i])

        assert final_error < initial_error
        


def test_weights_change_when_predictions_wrong():
    input = .65
    starting_weights = [.3, .2, .9]
    trues = [0, 1, 0]

    final_weights, error_history, weight_history = gradient_descent_outputs(input, starting_weights.copy(), trues, .1, 1)

    for i in range(3):
        assert final_weights[i] != starting_weights[i]


if __name__ == "__main__":

    tests = [
        test_predictions_move_closer_to_targets,
        test_weights_change_when_predictions_wrong
    ]

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError:
            print(f"FAIL: {test.__name__}")