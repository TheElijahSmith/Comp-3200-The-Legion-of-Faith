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

# The two layer neutral network
def neural_network(input, weights):
  hidden = vect_mat_mul(input, weights[0])
  prediction = vect_mat_mul(hidden, weights[1])

  return hidden, prediction


blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]


input = [8.5, 0.65, 1.2]

# This is going to be the hidden layer!
ih_wgt = [[0.1, 0.2, -0.1,],
          [-0.1, 0.1, 0.9],
          [0.1, 0.4, 0.1]]



# This is going to me the predction layer!
hp_wgt = [[0.3, 1.1, -0.3],
          [0.1, 0.2,  0.0],
          [0.0, 1.3, 0.1]]



weights = [ih_wgt, hp_wgt]

# This will run the network on all four sparring senses
for i in range(4):
  input = [blade_angle[i], balance[i], breath[i]]

  hidden, prediction = neural_network(input, weights)

  print("Sensing", i)
  print("Hidden Values:", hidden)
  print("Final prediction:", prediction)
                                      

# Answering the Question
# The hidden values are created by the first layer of the neural netowrk.
# Each hidden value combines the three weights.
#
# I would say for the middle lyaer to be useful it would need an activation function.
# So it can learn more complex relationships




#  This is going to be the Numpy version

blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

# The input is now going to be the hidden weights
ih_wgt = np.array([[0.1, 0.2, -0.1],
                   [-0.1, 0.1, 0.9],
                   [0.1, 0.4, 0.1]])



# This is going to be the prediction weights
hp_wgt = np.array([[0.3, 1.1, -0.3],
                   [0.1, 0.2, 0.0],
                   [0.0, 1.3, 0.1]])

# This is going to run all four sparring sensings
for i in range(4):
  input = np.array([blade_angle[i], balance[i], breath[i]])

  # Calculate the hidden layer by using dot product
  hidden = input.dot(ih_wgt.T)

  # Calculate the final prediction using the hidden layer
  prediction = hidden.dot(hp_wgt.T)

  print("Numpy Sensing", i)
  print("Numpy Hidden Values:", hidden)
  print("Numpy Final Predictions:", prediction)










