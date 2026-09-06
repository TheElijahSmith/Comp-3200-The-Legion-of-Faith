# week3/part1_error.py
import numpy as np

# No Numpy version

balance = [0.65, 0.80, 0.80, 0.90]
clean = [1, 1, 0, 1]
weight = 0.5

def squared_error(prediction, goal):
    # Subtracts goal from prediction and square it to find error
    return (prediction - goal) ** 2

def mean_squared_error(predictions, goals):
    #find each squared error, add them up, and divide by the count
    for input in balance:
        count = 0
        for p, g in zip(predictions, goals):
            count += squared_error(p, g)
        return count / len(predictions)

# Numpy Version

def np_squared_error(prediction, goal):
    # np.square will perform elementwise squaring
    return np.square(prediction - goal)

def np_mean_squared_error(predictions, goals):
    # Convert lists to arrays so numpy can work
    preds = np.array(predictions)
    gls = np.array(goals)

    # np.mean and np.square to get avg for all squared errors
    return np.mean(np.square(preds - gls))

# Verify both versions agree

# compute the squared errors of numpy and non-numpy
plain_se = [squared_error(p, g) for p, g in zip(balance, clean)]
numpy_se = [np_squared_error(p, g) for p, g in zip(balance, clean)]

# compute mean squared errors of numpy and non-numpy
plain_mse = mean_squared_error(balance, clean)
numpy_mse = np_mean_squared_error(balance, clean)

# a = regular version
# b = numpy version
# .#f = num of decimal places
print("Squared Errors (plain vs numpy):")
for i, (a, b) in enumerate(zip(plain_se, numpy_se)):
    # Print each sensing side-by-side
    # abs(a - b) < 1e-10 checks “close enough” instead of exact equality
    print(f"Sensing {i+1}: {a:.6f}   {b:.6f}   match={abs(a - b) < 1e-10}")

print("\nMean Squared Error:")
# Same comparison for the overall mean squared error
print(f"plain={plain_mse:.8f}   numpy={numpy_mse:.8f}   match={abs(plain_mse - numpy_mse) < 1e-10}")

# The results do happen t match in this scenario. The reason we don't want to lean on this is becuase 
# these are all very small numbers and a very small data set in terms of size. Numpy is also able to 
# add the terms of a sum in different orders than my loop in the regular version. Changing the order 
# does risk the final numbers differing by any value depending on how severe the order might be.