"""
weight = .1


def neural_network(input, weight):

    pred = input * weight
    return pred

blade_angle = [8.5, 9.5, 9.9, 9.0]
input = blade_angle[0]
pred = neural_network(input, weight)
print(pred)

for i in range(len(blade_angle)):
    pred = neural_network(blade_angle[i], weight)
    print(f"sensing {i}: input={blade_angle[i]} prediction={pred}")
"""

# ---- Weighted sum: pair-multiply, then add ----
def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i] # multiply and accumulate
    return output

# ---- Network with multiple inputs ----
weights = [0.1, 0.2, 0.0] # one weight per input

def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred

# ---- One sparring sensing: three readings ----
blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

input = [blade_angle[0], balance[0], breath[0]]
pred = neural_network(input, weights)
print(pred) # 0.9800000000000001