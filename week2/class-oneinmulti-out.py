def ele_mul(number, vector):
    output = [0] * len(vector)
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

weights = [0.3,.2,.9]

def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred

# ---- One sensing of balance ----
balance = [0.65, 0.80, 0.80, 0.90]
input = balance[0]
pred = neural_network(input, weights)
print(pred) # [0.195, 0.13, 0.5850000000000001]

def vect_mat_mul(vector, matrix):
    output = [0] * len(matrix)
    for i in range(len(matrix)):
        output[i] = w_sum(vector,
                          matrix[i])
    return output

# angle balance breath
weights = [[0.1, 0.1, -0.3], # opens left?
            [0.1, 0.2, 0.0], # strikes high?
            [0.0, 1.3, 0.1]] # feints?

input = [8.5, .65, 1.2]
pred = vect_mat_mul(input, weights)
print(pred) # [0.555, 0.9800000000000001, 0.9650000000000001]

 # ---- Two weight matrices ----
# angle balance breath
ih_wgt = [[0.1, 0.2, -0.1], # -> hid[0]
            [-0.1, 0.1, 0.9], # -> hid[1]
            [0.1, 0.4, 0.1]] # -> hid[2]

# hid0 hid1 hid2
hp_wgt = [[0.3, 1.1, -0.3], # -> opens left?
            [0.1, 0.2, 0.0], # -> strikes high?
            [0.0, 1.3, 0.1]] # -> feints?

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    hid = vect_mat_mul(input, weights[0]) # layer 1
    pred = vect_mat_mul(hid, weights[1]) # layer 2
    return pred

input = [8.5, 0.65, 1.2]
pred = neural_network(input, weights)
print(pred) # [0.21350000000000002, 0.14500000000000002, 0.5065]