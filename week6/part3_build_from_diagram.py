import numpy as np

# Relu function
def relu(x):
    return np.maximum(0, x)

# Deriv of Relu
def relu_deriv(y):
    return (y > 0).astype(float)

# Train from digram function to run through our layers to reach a (1, 1) matrix on layer 3
def train_from_diagram(tells, strike, alpha, epochs, seed):
    # seed for the following program
    np.random.seed(seed)
    
    # Diagram Shapes
    weights_0_1 = 2 * np.random.random((3, 8)) - 1
    weights_1_2 = 2 * np.random.random((8, 4)) - 1
    weights_2_3 = 2 * np.random.random((4, 1)) - 1

    error_history = []

    for epoch in range (epochs):

        for i in range(len(tells)):
            total_error = 0

            # running through the layers
            layer_0 = tells[i: i+1] # (1, 3)
            layer_1 = relu(layer_0 @ weights_0_1) # (1, 8)
            layer_2 = relu(layer_1 @ weights_1_2) # (1, 4)
            layer_3 = layer_2 @ weights_2_3 # (1, 1)

            # increase total error on each runthrough
            layer_3_delta = layer_3 - strike[i:i+1]
            total_error += np.sum(layer_3_delta ** 2)

            # Backpropogation (Find delta for each layer)
            layer_2_delta = (layer_3_delta @ weights_2_3.T) * relu_deriv(layer_2)
            layer_1_delta = (layer_2_delta @ weights_1_2.T) * relu_deriv(layer_1)

            # update weights
            weights_2_3 -= alpha * layer_2.T @ layer_3_delta
            weights_1_2 -= alpha * layer_1.T @ layer_2_delta
            weights_0_1 -= alpha * layer_0.T @ layer_1_delta

        # Update error history for this iteration
        error_history.append(total_error)

        # Print total error every 30 epochs
        if epoch % 30 == 0:
            print(f"Epoch {epoch}: Total Error = {total_error:.10f}")

    # Print final pred vs goals

    print("\n Final Predictions vs. Goals")
    for i in range(len(tells)):
        layer_0 = tells[i: i+1]
        layer_1 = relu(layer_0 @ weights_0_1)
        layer_2 = relu(layer_1 @ weights_1_2)
        layer_3 = layer_2 @ weights_2_3

        print(f"Prediction: {layer_3[0, 0]:.6f}, Goal: {strike[i]}")


    return error_history

if __name__ == "__main__":
    # Example Trial Dataset setup (matching Week 5)
    tells = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])
    strike = np.array([1, 1, 0, 0])

    # Run refactored loop
    history = train_from_diagram(tells, strike, alpha=0.1, epochs=150, seed=4)
    print(f"Final epoch error: {history[-1]:.10f}")

""" Comment Block
weights_0_1 connects layer_0 to layer_1.
It must be (3, 8) because layer_0 has 3 values
and layer_1 has 8 values.
 
weights_1_2 connects layer_1 to layer_2.
It must be (8, 4) because layer_1 has 8 values
and layer_2 has 4 values.
 
weights_2_3 connects layer_2 to layer_3.
It must be (4, 1) because layer_2 has 4 values
and layer_3 has 1 value.
"""

### FOR PART 5 OF ASSINGNMENT ###
# the diagram discipline might become the most important skill we learn this semester because diagrams set 
# up the framework for what we will be working on. If I see a diagram of a model, I should be able to figure 
# out what type of code I will need just by looking at the model. This is definitely not an easy skill but 
# it is one that is necessary. For me, it was difficult to write the network I have never seen before. I 
# struggled a lot and took to referring to other models from our notes to fill in gaps I had in my logic. 
# It was far from simple but by looking back, I did have reference points in my previous code to figure 
# things out. I think what Korrith's disciples lost was the ability to see the full picture of what they 
# are building. Without the diagram, you don't necessarily know where you are going to start and stop in a 
# given program unless it is explicitly told to you and gien time you may never get that.