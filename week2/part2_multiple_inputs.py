import numpy as np

# Weighted sum function, takes input vector (a) and
# computes the dot product with weight vector (b)
# Returns weighted_sum (A.K.A. the dot product)
def w_sum(a, b):
    assert len(a) == len(b) #makes sure the vectors are the same length
    weighted_sum = 0 # init w_sum total to 0
    for i in range(len(a)):
        #For all of the inputs and weights, multiply the input at the i'th
        #index with the weight at the same index and add that to the w_sum total
        weighted_sum += a[i]*b[i]
    return weighted_sum

# Uses w_sum func to get the weighted sum of an input vector and weight vector
#returns the weighted sum
def neural_network(inputs, weights):
    pred = w_sum(inputs, weights)
    return pred

blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
balance = [0.65, 0.80, 0.80, 0.90] # balance reading
breath = [1.2, 1.3, 0.5, 1.0] # exhalations per second

weights = [0.1, 0.2, 0.0]

sessions = [0 for _ in range(len(blade_angle))] #Init a vector that is the size of
# the total number of sessions with a placeholder val 0 to store each session's vals.

for i in range(len(blade_angle)):
    # Set each session's values to the corresponding session's values.
    sessions[i] = [blade_angle[i], balance[i], breath[i]]

scratch_preds = [0 for _ in range(len(sessions))] # Create vector of zeros to hold scratch
# preds for comparison with NumPy

for i in range(len(sessions)):
    # For each session, predict off of its values and the weights and store the pred in
    # scratch_preds
    pred = neural_network(sessions[i], weights)
    scratch_preds[i] = pred 


#---------------NumPy version---------------

# NumPy needs arrays to be np.arrays to be able to be used with the special np funcs.
np_weights = np.array(weights) #make np array of weights
np_sessions = np.array(sessions) #make equivalent np array of sessions

def np_neural_network(input, weights):
    # NumPy's built in .dot function acts just as the scratch w_sum func.
    pred = input.dot(weights)
    return pred

np_preds = np.zeros(len(sessions)) # Create vector of zeros to store NumPy preds for comparison.
#np.zeros creates a vector of zeros with the specified dimensions 

for i in range(len(sessions)):
    # Get the np_pred using the np_neural_network for every session and the weights
    # and store the pred in np_preds.
    np_pred = np_neural_network(np_sessions[i], np_weights)
    np_preds[i] = np_pred

print("\nPrediction comparisons:")
for i in range(len(sessions)):
    print(f"Session {i} predictions: scratch = {scratch_preds[i]:.4f},  NumPy = {np_preds[i]:.4f}")


if np.allclose(scratch_preds, np_preds):
    # If the preds are close within a very small tolerance, print that they match,
    # otherwise print that they don't
    print("Preds match!")
else: print("Preds do not match!")

"""
The preds match!
If there were ten thousand terms, the preds would probably not be an exact match
because NumPy can perform the dot prod operation in any order it would like, which
would probably cause the floating point number to be rounded every so slightly differently
in the end.
"""