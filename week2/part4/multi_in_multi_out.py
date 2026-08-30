import numpy as np


# Weighted sum function from part 2
# Weighted sum function, takes input vector (a) and
# computes the dot product with weight vector (b)
# Returns weighted_sum (A.K.A. the dot product)
def w_sum(a, b):
    assert len(a) == len(b) #makes sure the vectors are the same length
    weighted_sum = 0 # init w_sum total to 0
    for i in range(len(a)):
        # For all of the inputs and weights, multiply the input at the i'th
        # index with the weight at the same index and add that to the w_sum total
        weighted_sum += a[i]*b[i]
    return weighted_sum


# Function that does dot product on each vector in a matrix
def vect_mat_mul(vect, matrix):
    output = [0] * len(matrix)
    for i in range(len(matrix)):
        output[i] = w_sum(vect, matrix[i])
    return output

#         angle balance breath
weights = [[0.1, 0.1, -0.3], # opens_left?
           [0.1, 0.2, 0.0], # strikes_high?
           [0.0, 1.3, 0.1]] # feints?

input = [8.5, 0.65, 1.2]

prediction = vect_mat_mul(input, weights)

# prints the predictions but typically has miniscule floating point errors
print(prediction)



# ------- Numpy Version -------- #


# Numpy allows dot product through a matrix to be calculated in one simple function
def neural_network(input, weights):
    # .dot(weights.T) runs all three dot products at once
    npPrediction = input.dot(weights.T) # .T = transpose, flips the rows and columns
    return npPrediction

#                   angle balance breath
weights = np.array([[0.1, 0.1, -0.3], # opens left?
                    [0.1, 0.2, 0.0], # strikes high?
                    [0.0, 1.3, 0.1]]) # feints?

input = np.array([8.5, 0.65, 1.2])

# weights.shape will print (3,3) to show that it is a 2d array with 3 rows and 3 columns
print("Weights dimensions: ", weights.shape)
# input.shape will print (3,) to show that it is a 1d array with 3 indexes
print("Input dimensions: ", input.shape)

npPrediction = neural_network(input, weights)

# no floating point errors since NumPy handles it within the .dot function
print(npPrediction)

# Prints True if the two values are within a certain range of each other to account for floating point errors
print(np.allclose(prediction, npPrediction, atol=1e-9))