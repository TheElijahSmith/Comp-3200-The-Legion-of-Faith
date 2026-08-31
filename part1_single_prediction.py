# List of blade angle values from the sparring sessions
blade_angle = [8.5, 9.5, 9.9, 9.0]
# Starting weight value to control prediction size
weight = 0.5

def neural_network(input, weight):
    # Multiply the single input number by the weight number
    return input * weight

# Loop through each blade angle reading
for angle in blade_angle:
    # Get prediction for current blade angle
    pred = neural_network(angle, weight)
    print(f"Blade Angle: {angle} | Prediction: {pred}")

"""
ANSWER TO QUESTIONS

1. What does the weight do?
   Controls how much influence an input feature has on the output prediction.

2. What if weight changed from 0.5 to 2.0?
   Amplifies the input signal, doubling the output score and increasing sensitivity.

3. What if weight is negative?
   Inverts the signal, causing higher inputs to produce lower/negative predictions.
"""
