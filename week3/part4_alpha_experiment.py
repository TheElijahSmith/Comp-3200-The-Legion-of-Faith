from part3_alpha import gradient_descent_alpha

def detect_divergence(errors):
  return errors[-1] > 0.99 * errors[0]

alpha_values = [0.001, 0.01, 0.1, 0.4, 0.5, 0.6, 1.0]

input = 2.0
goal = 0.8
weight = 0.5
iterations = 20

print("Alpha Experiment Results")
print("---------------------------")
print("Alpha\tFinal Error\tDivergence")

for alpha in alpha_values:
  errors = gradient_descent_alpha(input, goal, weight, alpha, iterations)

  final_error = errors[-1]
  diverged = detect_divergence(errors)

  print(f"{alpha}\t{final_error:.6f}\t{diverged}")


## Observations:
## The largest alpha that still converges is 0.4
## The smallest Alpha that diverges is 0.6
## The true stability boundary is alpha = 0.5
## At alpha = 0.5, the error stays exactly 0.04, so it is 
## Neither convering nor diverging. It is the stability boundary.
## Smaller alpha value are more stable but take longer to converge.
## Larger alpha values can converge faster, but if alpha is too large,
## the error grows and the training diverges
