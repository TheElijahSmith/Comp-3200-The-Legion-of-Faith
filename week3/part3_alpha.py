import numpy as np
from part1_error import squared_error
from part2_gradient_descent import gradient_descent


def gradient_descent_alpha(input, goal, weight, alpha, iterations):

    print("Gradient Descent Alpha")

    # List of the error at each iterations
    error_list = []

    # Loop that repeats given a number of iterations
    for n in range(iterations):
        # Make the predictions by multiplying the weight and the input
        prediction = input * weight

        # Compare how wrong we were
        error = squared_error(prediction, goal)
        error_list.append(error)

        # Delta is the difference between our guess and the goal
        delta = prediction - goal

        # Weight_delta is the adjustment we need to make to the weight
        weight_delta = input * delta 

        # Update the weight
        weight = weight - alpha * weight_delta

        # Print the clean iteration log
        print(f"Iteration {n + 1}: Error = {error:.4f} | Prediction = {prediction:.4f} | Current weight = {weight:.4f}")

    return error_list



# gradient_descent(2, .8, .5, 20)

# Gradient descent explodes because gradients are multiplied each layer and jumps out of the curve, but 
# by introducing the alpha it dampens the explosion by mutiplying a decimal less than one.

# gradient_descent_alpha(2, .8, .5, .1, 20)


# --------------------------------------------------- NUMPY SECTION -----------------------------------------------------------------

def gradient_descent__alpha_numpy(input, goal, weight, alpha, iterations):

    print("Gradient Descent Alpha Numpy")

    # Converting the variables to np.float64
    np_input = np.float64(input)
    np_goal = np.float64(goal)
    np_weight = np.float64(weight)

    error_list = []

    for i in range(iterations):
        np_prediction = np_input * np_weight
        
        # Using np.square to calculate the squared error
        np_error = np.square(np_prediction - np_goal)
        error_list.append(np_error)

        np_delta = np_prediction - np_goal
        np_weight_delta = np_input * np_delta

        # Update the weight
        np_weight = np_weight - alpha * np_weight_delta

    return error_list



blade_angle = [8.5, 9.5, 9.9, 9.0]

for i in blade_angle:
    gradient_descent_alpha(i, .8, .5, .01875, 20)

# The largest alpha I could find that settles down at our goal was .01875. Anything more only got to the goal
# on the last iteration which did not feel like it was certain enough for me to prove that it worked.