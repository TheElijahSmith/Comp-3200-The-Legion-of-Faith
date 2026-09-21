import numpy as np
np.random.seed(1)
def relu(x): return (x > 0) * x # max(0, x) elementwise

### Initial Dataset ###
# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
                    [0, 1, 1], # no shift, guard drop, exhale
                    [0, 0, 1], # only exhale
                    [1, 1, 1]]) # all three (the bluff)
# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)
# it does not appear that strike is used in this code



def forward(layer_0, weights_0_1, weights_1_2):
    # First weighted sum, followed by ReLU
    layer_1 = relu(layer_0.dot(weights_0_1)) # shape (1, 4)
    # Second weighted sum, no output activation yet
    layer_2 = layer_1.dot(weights_1_2) # shape (1, 1)
    return (layer_1, layer_2)

# Initializing our weights
hidden_size = 4
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 * 4
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 * 1

# Pass all 4 sensings through forward() and print the results
for i, sensing in enumerate(tells): # No longer initializing layer 0. 
    # sensing in enumerate(tells) allows for the properinitialization 
    # of layer 0 without ignoring layers 3 and 4. Previous attempts 
    # where I defined layer 0 would lead to the program ignoring layers 3 and 4
    
    layer_1, layer_2 = forward(sensing, weights_0_1, weights_1_2)

    print(f"Sensing {i + 1}:") # Prints and Sensing 1, Sensing 2, etc.
    # Print each layer of the Sensing
    print("Layer 1:", layer_1)
    print("Layer 2:", layer_2)
    # Empty space to organize by Sensing
    print()

## Comment Block

"""
Layer 0 is a (1, 3) matrix that shows the row it is currently viewing in tells.
Layer 1 is a (1, 4) matrix that holds 4 spearate values as a result of multiplying 
the first row in tells by the 3*4 matrix from the weights. The layer 2 matrix is 
a 1*1 matrix resulting from multiplying layer 1 with the 4*1 matrix from the weights.

To plug things in on how we get from layer 0 to layer 2 with the weight matrices, we would start with 
(1, 3) * (3, 4). We can cancel out the middle 3s and we end up with a (1, 4) matrix which is what layer 
1 is the (3, 4) matrix is the weights_0_1 in this equation. The reason we canel out the middle numbers is 
because they must match to perform dot multiplication and the resulting Matrix is always the outer values. 
From this, we do (1, 4) * (4, 1). (4, 1) is the other weight matrix. Canceling out the 4s, we end up with 
a (1, 1) matrix which is what layer 2 is.
"""