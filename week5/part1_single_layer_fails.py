import numpy as np
# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
    [0, 1, 1], # no shift, guard drop, exhale
    [0, 0, 1], # only exhale
    [1, 1, 1]]) # all three (the bluff)
# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

SEED = 42



# Train single layer network
def single_layer_train(tells, strike, alpha, epochs, seed):
    # set seed for reproducibility
    np.random.seed(seed)

    # init a vector of weights of size (3,)
    weights = 3 * np.random.random(3)

    # for every epoch
    for i in range(epochs):
        # for every row in the tells vector
        for j in range(len(tells)):
            # get pred for this round
            pred = tells[j] @ weights

            # calculate the delta
            delta = pred - strike[j]

            # calculate weight deltas (delta * input)
            weight_deltas = delta * tells[j]

            # new weights = old weights - alpha * weight_deltas
            weights -= alpha * weight_deltas

    print(" Foot shift, guard drop, exhale ")
    print(weights)

    # init total error
    total_error = 0

    # Printing final four preds vs goals; also calculate errors
    for i in range(len(tells)):
        pred = tells[i] @ weights.T
        print(f"Pred = {pred}, goal was {strike[i]}")

        #error = delta ^ 2
        error = (pred - strike[i]) ** 2
        total_error += error

    return total_error


        


    
    


"""
This fails because, for this data, there is not a single input that is telling the story on its own.
Looking at the columns in tells every input has an equivalent number of strikes to holds when present
and when not present (i.e. foot shift when present had a strike one time and hold the other, and the same
thing when not present; this goes for guard drop, too, and exhale is just always present so it
means nothing valuable.) The valuable relationship is the XOR relationship between foot strike and guard
drop which is NOT representable through a simple linear weighted sum.
"""




if __name__ == '__main__':
    single_layer_train(tells, strike, .1, 60, SEED)