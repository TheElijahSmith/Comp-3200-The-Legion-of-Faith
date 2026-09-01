knob_weight = 0.5 # the network's "attention" knob
input = 0.5 # one sparring impression
goal_pred = 0.8 # what the padawan should have sensed

# 1. Predict (forward propagation)
pred = input * knob_weight # 0.5 * 0.5 = 0.25

# 2. Compare -- put a number on the miss
error = (pred - goal_pred) ** 2 # (0.25 - 0.8)^2 = 0.3025
print(f"{error:.4f}") # 0.302

weight = .5
input = .5


#----------------One Step of Adjustment---------

# ---- One full Predict -> Compare -> Learn step ----
weight = 0.1
input = 8.5
goal_pred = 1.0 # truth: a clean strike

# 1. PREDICT
pred = input * weight # 8.5 * 0.1 = 0.85

# 2. COMPARE
error = (pred - goal_pred) ** 2 # (0.85 - 1.0)^2 = 0.0225
delta = pred - goal_pred # 0.85 - 1.0 = -0.15

# 3. LEARN
weight_delta = delta * input # -0.15 * 8.5 = -1.275
weight = weight - weight_delta # 0.1 - (-1.275) = 1.375
print(f"new weight: {weight:.4f}") # new weight: 1.375


#----------------Several Steps-----------

# Same loop, run four times
weight = 0.0 # blank slate
goal_pred = 0.8
input = 1.1

for iteration in range(4):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input
    weight = weight - weight_delta

print("Error:" + str(round(error, 4))
    + " Pred:" + str(round(pred, 4)))

#---------------From Scratch --> NumPy
weight = 0.0
goal_pred = 0.8
input = 1.1

for iteration in range(4):
    # PREDICT
    pred = input * weight
    # COMPARE
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    # LEARN
    weight_delta = delta * input
    weight = weight - weight_delta

    print("Error:" + str(error)
        + " Pred:" + str(pred))




#---------NUMPY
import numpy as np

weight = np.float64(0.0)
goal_pred = np.float64(0.8)
input = np.float64(1.1)

for iteration in range(4):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input
    weight = weight - weight_delta

    print(f"Error:{error:.4f}"
        + f" Pred:{pred:.4f}")