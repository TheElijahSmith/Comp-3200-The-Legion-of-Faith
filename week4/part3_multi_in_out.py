import numpy as np
from helpers import vect_mal_mul



def outer_prod(a, b):
  result = []
for i in range(len)(a)):
    row = []
    for j in range (len(b)):
      row.append(a[i] * b[j])
    result.append(row)

return result



def gradient_descent_full(input, weights, trues, alpha, iterations):
  weights = [row[:] for row in weights]
  
  error_history = []
  weight_history = []
  prediction_history = []

  for i in range(iterations):
    predictions = vect_mat_mul(input, weights)
    prediction_history.append(predictions[:])
    deltas = []
    squared_errors = []

   for j in range(len(predictions)):
     delta = predictions[j] - trues[j]
     deltas.append(delta)
     squared_errors.append(delta ** 2)
   error = sum(squared_errors) / len (squared_errors)
   error_history.append(error)
   weight_deltas = outer_prod(deltas, input)

   for r in range(len(weights)):
     for c in range(len(weights[r])):
       weights[r][c] = weights[r][c] - alpha * weight_deltas[r][c]
       
   weight_snapshot = [row[:] for row in weights]
   weight_history.append(weight_snapshot)
  return weights, error_history, weight_history, prediction_history

def gradient_descent_full_numpy(input, weights, trues, alpha, iterations):
  input = np.array(input, dtype=float)
  weights = np.array(weights, dtype=float)
  trues = np.array(trues, dtype=float)

  error_history = []
  weight_history = []
  prediction_history = []

  for i in range(iterations):
    predictions = np.dot(input, weights)
    prediction_history.append(predictions.copy())

    deltas = predictions - trues
    error = np.mean(deltas ** 2)
    error_history.append(error)

    weight_deltas = np.outer(deltas, input)
    weights = weights - alpha * weight_deltas

    weight_history.append(weights.copy())

  return weights, error_history, weight_history, prediction_history


if __name__ == '__main__':
  weights = [[0.1, 0.1, -0.3],
             [0.1, 0.2, 0.0],
             [0.0, 1.3, 0.1]]
  input = [8.5, 0.65, 1.2]
  trues = [0.0, 1.0, 0.0]

  final_weights, error_history, weight_history, prediction_history = gradient_descent_full( input, weights. trues, 0.01, 15 )

  print("\nIteration log:")

  for i in range(15):
    print( 
      f"Iteration {i + 1}: "
      f"Prediction = {prediction_history[i]} | "
      f"Error = {error_history[i]:.4f} | "
      f"Weights = {weight_history[i]}"
    )
        

  print("Final weights:")
  print(final_weights)

  print("Initial error:", error_history[0])
  print("Final error:", error_history[-1])

  numpy_weights, numpy_errors, numpy_weight_history, numpy_predictions = gradient_descent_full_numpy( input, weights, trues, 0.01, 15)

  print("\nNumpy final weights:")
  print(numpy_weights)

  print("\nParity check:")
  print("Weights match:", np.allclose(final_weights, numpy_weights))
  print("Errors match:", np.allclose(error_history, numpy_errors))

  print("\nWeight movement:")

  initial_weights = [[0.1, 0.1, -0.3],
                     [0.1, 0.2, 0.0],
                     [0.0, 1.3, 0.1]]

  for r in range(len(final_weights)):
    for c in range(len(final_weights[r])):
      movement = abs(final_weights[r][c] - initial_weights[r][c])
      print(f"Weight [{r}][{c}] moved by {movement:.6f}"}

# Weight movement depends on both the output delta and the input value.
# Each weight is corrected using:
# weight_delta[i][j] = delta[i] * input[j]
# Within one row, larger input values cause larger weight movement.
# between differemt rows, a larger output delta causes more mvement
# meaning its prediction is already close to the true target
                    
  
