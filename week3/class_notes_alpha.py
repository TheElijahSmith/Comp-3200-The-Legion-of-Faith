#------------FROM SCRATCH---------

weight = 0.5
goal_pred = 0.8
input = 2.0
alpha = 0.1

for iteration in range(20):
    # 1. PREDICT
    pred = input * weight
    # 2. COMPARE
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    # 3. LEARN
    weight_delta = delta * input
    weight -= alpha * weight_delta

    print(f"Iter {iteration}: "
    + f"Pred={pred:.4f}")
    + f"Error={error:.4f} "

#----------------NUMPY---------

import numpy as np

weight = np.float64(0.5)
goal_pred = np.float64(0.8)
input = np.float64(2.0)
alpha = 0.1

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input
    weight -= alpha * weight_delta

    print(f"Iter {iteration}: "
    + f"Error={error:.6f} "
    + f"Pred={pred:.4f}")