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

