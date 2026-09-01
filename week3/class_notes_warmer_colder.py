# ---- Warmer, colder: try both directions every step ----
weight = 0.5
input = 0.5
goal_prediction = 0.8
step_amount = 0.001 # how big a nudge each iteration

for iteration in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2
    print("Error:" + str(error) + " Pred:" + str(prediction))

    # Try a slightly larger weight
    up_prediction = input * (weight + step_amount)
    up_error = (up_prediction - goal_prediction) ** 2

    # Try a slightly smaller weight
    down_prediction = input * (weight - step_amount)
    down_error = (down_prediction - goal_prediction) ** 2

    # Keep whichever direction was warmer (lower error)
    if down_error < up_error:
        weight = weight - step_amount # down was warmer; go down
    if down_error > up_error:
        weight = weight + step_amount # up was warmer; go up