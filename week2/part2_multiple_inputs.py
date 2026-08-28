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

for i in range(len(blade_angle)):
    # For each sparring session, set the current session into a vector
    # and predict off the session and the weights and print the pred
    current_session = [blade_angle[i], balance[i], breath[i]]
    session_pred = neural_network(current_session, weights)
    print(f"Round {i} prediction {session_pred:.3f}")

