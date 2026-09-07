# test_learning.py -- Unit tests for the Week 3 learning loop
from part1_error import squared_error, mean_squared_error
from part2_gradient_descent import gradient_descent
from part3_alpha import gradient_descent_alpha, gradient_descent__alpha_numpy
from part4_alpha_experiment import detect_divergence
import numpy as np

def test_squared_error():
    print("==========Testing Squared Error===========")
    #squared_error(pred, goal); performs (pred-goal)^2
    assert abs(squared_error(5, 10) - 25) < 1e-10, "A delta of -5 should result in 25 when squared"
    assert abs(squared_error(1, .8) - .04) < 1e-10, ".2 squared should result in .04"
    assert abs(squared_error(.35, .35)) < 1e-10, "Perfect pred should = 0"

def test_gradient_descent_convergence():
    print("==========Testing Gradient Descent Convergence===========")
    # gradient_descent(input, goal, weight, iters)
    # Checks if the last error in the list of errors is less than .1
    #    (.05 means our final pred was ~.223 off of the goal because the error is delta squared)
    # All of these should converge because the alpha (in this case 1) is easily between 0 and 2/input^2
    #   They would run into a problem if the input was >= sqrt(2)
    assert gradient_descent(.9, .3, .8, 20)[-1] < .05, "If this converges, in 20 iters the final delta should be <.223"
    assert gradient_descent(.3, 1, .2, 20)[-1] <.05, "Same as above"
    assert gradient_descent(.6, -.4, .5, 20)[-1] <.05, "Same as above"

def test_alpha():
    print("==========Testing Alpha===========")
    #gradient_descent_alpha(input, goal, weight, alpha, iters)
    # If the second delta is > than the first it will diverge.
    #   As stated above, the alpha must be between 0 and 2/input^2 to to get convergence
    assert (
        #This should diverge, 2/2^2 = .5 (the alpha val must be less than this), and the alpha is 1. 1>.5
        gradient_descent_alpha(2, 1, 1, 1, 20)[-1] > 1000 
        and 
        # .3 < .5
        gradient_descent_alpha(2, 1, 1, .3, 20)[-1] < .01
    ), "First running should diverge and second should converge."

    assert (
        #For these, the maximum alpha to guarantee convergence is < 2/11^2 = approx .0118
        gradient_descent_alpha(13, -9, .1, 1, 20)[-1] > 1000
        and
        gradient_descent_alpha(13, -9, .1, .005, 20)[-1] < .01
    ), "Same as above"

def test_divergene_detection():
    #detect_divergence(errors). Should return true if the set is diverging
    #tests big divergences and small divergences as well as oscillation, and converging sets
    diverging_errors_big = [1,2,4,8,16,32,64,128]
    diverging_errors_small = [.1, .11, .13, .15, .17, .2, .25]
    diverging_errors_oscillating = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    converging_errors_big = [10000, 9000, 8400, 7800, 7400, 7000, 6500, 6000]
    converging_errors_small = [.1,.097,.093,.088,.085,.082,.08,.075]

    assert(
        detect_divergence(diverging_errors_big) and detect_divergence(diverging_errors_small)
        and detect_divergence(diverging_errors_oscillating)
        ), "Should detect diverging errors"

    assert(
        not detect_divergence(converging_errors_big) and not detect_divergence(converging_errors_small)
    ), "Should detect convergence"


def test_scratch_numpy_agreement():
    # The final errors for the scratch and numpy gradient descents should differ only by floating point rounding
    final_scratch_e = gradient_descent_alpha(13, -9, .1, .005, 20)[-1]
    final_numpy_e = gradient_descent__alpha_numpy(13, -9, .1, .005, 20)[-1]
    assert abs(final_scratch_e-final_numpy_e) < 1e-10, "Should only differ by floating point rounding"
    



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