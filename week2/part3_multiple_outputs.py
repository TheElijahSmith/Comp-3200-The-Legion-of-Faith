import numpy as np

# used ai to debug code and understand the numpy section

def ele_mul(scalar, vector): # Function used to scale each weight by an input value
    output = [0] * len(vector)
    for i in range(len(vector)):
        output[i] = scalar * vector[i]
    return output

weights = [0.3, 0.2, 0.9] # 1 input with 3 outputs [open to the left, strike high, feint]
balance = [0.65, 0.80, 0.80, 0.90] # balance reading

# function will take an input and the weights to return 3 separate predictions
def neural_network(input, weights): 
    pred = ele_mul(input, weights)
    return pred

# function will do 1 senseing of balance
for input in balance: # cycles through all 4 elements in balance
    pred = neural_network(input, weights) # current element in cycle
    print(pred) # result of element n

print ("Numpy's Version")

# numpy requires arrays
weights_np = np.array(weights)

def np_neural_network(input, weights):
    pred = input * weights
    return pred

for input in balance:
    pred = np_neural_network(input, weights_np)
    print(pred)

# the usage of ele_mul is not needed for this version becuase numpy will do elementwise multiplication 
# when interacting with arrays. It essentially multiplies each element in the array by the input we give it, 
# removing the need for a loop. It allows us to remove everything from ele_mul from the code alltogether.