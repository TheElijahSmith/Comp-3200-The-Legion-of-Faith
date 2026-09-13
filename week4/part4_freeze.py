from part1_multi_input import ele_mul, w_sum

def gradient_descent_frozen(input, weights, true, alpha, iterations, frozen):
    # initialize error and weight history lists
    error_history = []
    weight_history = []

    # For every iteration:
    for i in range(iterations):
        #append current weights
        weight_history.append(weights)


        #calculate preds
        pred = w_sum(input, weights)

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
        for j in range(len(weights)):
            weights[j] = weights[j] - alpha * weight_deltas[j]

    final_weights = weights

    return final_weights, error_history, weight_history

sensing = [8.5, .65, 1.2]
starting_ws = [.1, .2, -.1]
true = 1

#For each test, these will be the different weights frozen, and the alphas for the corresponding test
frozen_indices = [[], [0, 2], [0,1]]
alphas = [.01, .3, .3]


        


