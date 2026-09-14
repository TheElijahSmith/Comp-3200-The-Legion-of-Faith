from part1_multi_input import ele_mul, w_sum

def gradient_descent_frozen(input, weights, true, alpha, iterations, frozen):
    # initialize error and weight history lists
    error_history = []
    weight_history = []
    ws = weights.copy()

    # For every iteration:
    for i in range(iterations):
        #append current weights
        weight_history.append(ws.copy())


        #calculate preds
        pred = w_sum(input, ws)

        #calculate delta and errors (pred - true) = delta
        delta = pred - true
        error = delta ** 2

        #add the error into the error history
        error_history.append(error)

        #Get weight_deltas by multiplying the delta * every input and storing in list
        weight_deltas = ele_mul(delta, input)

        #Frozen is a list of indices for weights that are to remain unchanged
        for index in frozen:
            #By setting the weight_deltas at that index to zero, the weight will not change
            #   weight - alpha(0) = weight
            weight_deltas[index] = 0

        #Update the weight vector by subtracting the corresponding weight delta * alpha for every weight
        for j in range(len(ws)):
            ws[j] = ws[j] - alpha * weight_deltas[j]

    final_weights = ws

    return final_weights, error_history, weight_history


# blade, balance, breath
sensing = [8.5, .65, 1.2]
starting_ws = [.1, .2, -.1]
true = 1

#For each test, these will be the different weights frozen, and the alphas for the corresponding test
frozen_indices = [[], [0, 2], [0,1]]
alphas = [.01, .3, .3]

iters = 5


for i in range(3):
    final_ws, errors, weights, = gradient_descent_frozen(sensing, starting_ws, true, alphas[i], iters, frozen_indices[i])

    print(f"\nTest {i+1}:")
    print(f"Final weights = {final_ws}")
    print(f"Weight history = {weights}")
    print(f"Errors = {errors}")


"""
In run 2, the balance weight is doing all of the correcting. In run 3, it is breath that does this.
When a weight is frozen, its point on the error curve is frozen, so the only way to move down the bowl
is for the other weights to translate the bowl towards the frozen weight, rather than moving the weight down
the bowl.

For run 1, S = 8.5^2 + .65^2 + 1.2^2 = 74.1125. 1 - .01*74.1125 = .258875
Run 2: S = .65^2 = .4225.    1 - .3 * .4225 = .87325
Run 3: S = 1.2^2 = 1.44.   1 -.3 *1.44 = .568

When blade angle is frozen, S is decreased by a very large amount (8.5^2 = 72.25). The miss must be multiplied by a number between
1 and -1, so 1 - alpha * S must be between 1 and -1. When S is very high (72.25) alpha must be much smaller to keep the miss from
being too large in magnitude, specifically alpha * S must be between 0 and 2. Alpha must be more than 0 and less than = 2/S.
2/74.1125 (S of first run) = ~.027, so alpha must be less than that. For the the other two runs, the greatest S = 1.44,
2/1.44 = ~ 1.389 is the max alpha for that run. So, the alpha can be much larger when your highest changeable inputs are much smaller.
"""


