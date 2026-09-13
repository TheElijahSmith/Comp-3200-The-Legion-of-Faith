def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i] # multiply and accumulate
    return output



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
