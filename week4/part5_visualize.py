import matplotlib
matplotlib.use("Agg") # render to a file; no display needed
import matplotlib.pyplot as plt

# training functions
from part1_multi_input import gradient_descent_multi
from part4_freeze import gradient_descent_frozen


## Figure 1 - Part 1's weights over time

# senses index 0 and alpha 0.01, for at least 10 iterations
weights, errors, weight_history = gradient_descent_multi(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 10) # sensing 0, weights, goal, alpha, iterations

for j, name in enumerate(["blade_angle", "balance", "breath"]):
    plt.plot([w[j] for w in weight_history], label=name)

plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 1 weights, alpha = 0.01")
plt.legend()
plt.savefig("week4/fig1_weights.png", dpi=150)
plt.close() # start a clean figure for the next plot

## Figure 2 - Part 4's Frozen vs Unfrozen function

# Unfrozen run
base_w, base_err, base_hist = gradient_descent_frozen(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 10, frozen=[] # same values as above and the inclusion of what is and isn't frozen
)

# Frozen = [0, 2] run balance is free
bal_w, bal_err, bal_hist = gradient_descent_frozen(
        [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 10, frozen=[0, 2] # same values as above and the inclusion of what is and isn't frozen
)

# Frozen = [0, 2] run breath is free
bre_w, bre_err, bre_hist = gradient_descent_frozen(
        [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 10, frozen=[0, 1] # same values as above and the inclusion of what is and isn't frozen
)

# Plot balance weight: Unfrozen vs Frozen=[0, 2]
plt.plot([w[1] for w in base_hist], label="balance unfrozen (alpha=0.01)")
plt.plot([w[1] for w in bal_hist], label = "balance frozen=[0,2] (alpha = 0.01)")

# Plot balance weight: Unfrozen vs Frozen=[0, 1]
plt.plot([w[2] for w in base_hist], label="balance unfrozen (alpha=0.01)")
plt.plot([w[2] for w in bre_hist], label = "balance frozen=[0,1] (alpha = 0.01)")

plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 4 frozen vs unfrozen")
plt.legend()
plt.savefig("week4/fig2_frozen.png", dpi=150)
plt.close() 

""" 
Comment Block:

1. The weights in the frozen examples are steeper than the unfrozen examples. Not by much but enough to see that the line has grown steeper.

2. The blade angle weight is the only one that does get steeper at a glance so it is the steepest by default. It does match the part 1 prediction.

3. They are not. They only appear to be because of how little the incline is. To see a more defined incline or decline, we could plot the error contribution or the absolute change per iteration.
"""
