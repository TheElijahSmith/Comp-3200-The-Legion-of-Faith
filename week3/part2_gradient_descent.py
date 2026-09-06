# Import the funcion of part 1 and numpy
from part1_error import squared_error
import numpy as np

def gradient_descent(sensing, goal, weight, iterations):
    # List of the error at each iterations
    error_list = []

    # Loop that repeats given a number of iterations
    for n in range(iterations):
        # Make the predictions multiplying the weight and the input/sensing
        prediction = sensing * weight
        # Compare how wrong we were
        error = squared_error(prediction, goal)
        error_list.append(error)
        # Delta make the difference between our guess and the goal
        delta = prediction - goal
        # Weight_delta is the math adjustment we need to make to the weight
        weight_delta = sensing * delta 
        # Update the weight
        weight = weight - weight_delta

        # Print the clean iteration log
        print(f"Iteration {n}: Error = {error:.4f} | Prediction = {prediction:.4f} | Current weight = {weight:.4f}")

    return error_list

def gradient_descent_numpy(sensing, goal, weight, iterations):
    # Convering the variables to np.float64
    np_sensing = np.float64(sensing)
    np_goal = np.float64(goal)
    np_weight = np.float64(weight)

    error_list = []

    for i in range(iterations):
        np_prediction = np_sensing * np_weight
        # Using np.square to calculate the squared error
        np_error = np.square(np_prediction - np_goal)
        error_list.append(np_error)

        np_delta = np_prediction - np_goal
        np_weight_delta = np_sensing * np_delta

        # Update the weight
        np_weight = np_weight - np_weight_delta

    return error_list

# This guard prevents the pritns from messing up
if __name__ == '__main__':
    # The small input balance dataset provided in the instrucions
    balance = [0.65, 0.80, 0.80, 0.90]
    clean = [1, 1, 0, 1]

    starting_weight = 0.5
    total_iterations = 20

    # Testing with the first sensing (Index 0)
    print('Training Sensing')
    gradient_descent(balance[0], clean[0], starting_weight, total_iterations)

    print('\n Training Sensing (NumPy version)')
    gradient_descent_numpy(balance[0], clean[0], starting_weight, total_iterations)
    print('The numbers match with the scratch version')

    # Testing with other sensings to see how the network behaves
    print('\n Training Sensing 1')
    gradient_descent(balance[1], clean[1], starting_weight, total_iterations)

    print('\n Training Sensing 2')
    gradient_descent(balance[2], clean[2], starting_weight, total_iterations)

# Does the same starting weight work well for all sensings? 
# Why o why not?
'''
No, the same starting weight of 0.5 work equall for all sensings.
Because if the sensing is 0 or is sensing 1, the goal is get 1,
so we start with a positive weight of 0.5 we are going on the right direction
sinse the begin, making the network learn quickly.
But if the sensing is 2, and the weight is 0.80 but the goal is 0, 
we are going in the opposite direction which makes the network learns slower
because our fisrt prediction was to high and wrong.
'''