import numpy as np

def relu(x):
    return (x > 0) * x

np.random.seed(42) # Callback to previous classes ;)

# Creating weights and initial matrix
X = np.random.randn(5, 4)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2)

# 2 layer lin output
Y_two = (X @ W1) @ W2

# creating the single collapsed matirx
W_prime = W1 @ W2
Y_one = X @ W_prime

# printing Y_two, and Y_one to show equality using allclose
print("\nnp.allclose(Y_two, Y_one): ")
print(np.allclose(Y_two, Y_one))

# Inserting form and printing it
Y_nl = relu(X @ W1) @ W2
print ("\nY_nl:")
print(Y_nl) 
print(" compared to Y_one")
print(Y_one)

## The collapse no longer holds becuase relu converts several value to 0. This ends up drastically 
# changing the results of the functions when it reaches it's final layer unlike Y_two compared to Y_one 
# which ends up showing equatity. There is no difference in calculations caused by forcing values to be 0.