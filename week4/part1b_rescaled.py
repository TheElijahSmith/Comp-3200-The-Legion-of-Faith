# import from the part1
from part1_multi_input import gradient_descent_multi

# the four sparring sensings, same as previous weeks
blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

def normalize(channel):
    # divide every value by the biggest value in that same list
    biggest = max(channel)
    return [value / biggest for value in channel]

# scale each channel against itself, never against another channel
blade_angle_norm = normalize(blade_angle)
balance_norm = normalize(balance)
breath_norm = normalize(breath)

print("blade_angle normalized:", blade_angle_norm)
print("balance normalized:    ", balance_norm)
print("breath normalized:     ", breath_norm)

# rebuild sensing 0 out of the normalized channels
sensing_0_scaled = [blade_angle_norm[0], balance_norm[0], breath_norm[0]]
print("\nsensing 0 raw:   ", [8.5, 0.65, 1.2])
print("sensing 0 scaled:", sensing_0_scaled)

starting_weights = [0.1, 0.2, -0.1]
goal = 1.0

# raw run, same as Part 1: alpha = 0.01
print("\n--- raw run, alpha=0.01 ---")
raw_weights, raw_errors, raw_hist = gradient_descent_multi(
    [8.5, 0.65, 1.2], starting_weights, goal, 0.01, 20
)

# scaled run, unchanged function, just bigger alpha since inputs are small now
print("\n--- scaled run, alpha=0.1 ---")
scaled_weights, scaled_errors, scaled_hist = gradient_descent_multi(
    sensing_0_scaled, starting_weights, goal, 0.1, 20
)

# print both error histories side by side
print("\nError comparison (raw vs scaled):")
print(f"{'iter':<6}{'raw error':<15}{'scaled error':<15}")
for i in range(len(raw_errors)):
    print(f"{i:<6}{raw_errors[i]:<15.8f}{scaled_errors[i]:<15.8f}")

'''
(a) Raw inputs at alpha=0.1 diverge, scaled inputs don't. Why:
weight_delta = delta * input. blade_angle = 8.5 is a big number, so any
delta gets multiplied into a big weight_delta, and with alpha=0.1 that
push is too large -- the weight overshoots the goal, which makes the next
delta bigger and flipped in sign, and it spirals out of control. Once every
input is scaled down to between 0 and 1, the same alpha=0.1 produces a
small weight_delta, so the updates stay controlled and the error goes down
steadily instead of exploding.

(b) The scaled run's error is bigger than the raw run's error at every
iteration, and that's not a regression. The two runs aren't measuring the
same thing -- the raw run's error is already tiny because its inputs and
weights happen to sit close to a solution, while the scaled run started
further from its solution in the new, smaller-scale space. What actually
matters when comparing runs is how fast and how stably each one converges,
not the raw error number at some fixed iteration, since that number
depends on the scale of the data.

(c) Min-max normalization is (value - min) / (max - min). For blade_angle,
the min is 8.5 (which is sensing 0's own value) and the max is 9.9, so:
(8.5 - 8.5) / (9.9 - 8.5) = 0 / 1.4 = 0
sensing 0's blade_angle becomes exactly 0. Since weight_delta = delta *
input, a 0 input means that weight's delta is 0 too, no matter what the
error is -- that weight stops updating completely. That's the same thing
that happens in Part 4 when a weight is frozen on purpose.
'''