import numpy as np


def gradient_descent_outputs(input, weights, trues, alpha, iterations):
    error_history = []
    weight_history = []

    for iteration in range(iterations):
        # Make predictions
        predictions = [input * weight for weight in weights]

        # Calculate deltas
        deltas = [predictions[i] - trues[i] for i in range(3)]

        # Calculate the Mean Squared Error
        errors = [d ** 2 for d in deltas]
        mse = sum(errors) / len(errors)

        error_history.append(mse)
        weight_history.append(weights.copy())

        # Calculate weight detlas
        weight_deltas = [delta * input for delta in deltas]
        for i in range(3):
            weights[i] -= alpha * weight_deltas[i]

        # Print stuff :)
        print("Iteration: ", (iteration + 1))
        print("Predictions: ", [round(p, 4) for p in predictions])
        print("Mean Squared Error: ", round(mse, 4))
        print("Weights: ", [round(w, 4) for w in weights])
        print()

    return weights, error_history, weight_history


# -- NumPy Version --
def gradient_descent_outputs_numpy(input, weights, trues, alpha, iterations):
    weights = np.array(weights)
    trues = np.array(trues)

    error_history = []
    weight_history = []

    for iteration in range(iterations):
        predictions = input * weights

        deltas = predictions - trues

        errors = deltas ** 2
        mse = np.mean(errors)

        error_history.append(mse)
        weight_history.append(weights.copy())

        weight_deltas = deltas * input

        weights -= alpha * weight_deltas

    return weights, error_history, weight_history


def main():
    weights = [0.3, 0.2, 0.9]
    input = .65
    trues_multi = [0, 1, 0]
    alpha = .1

    final_weights, error_history, weight_history = gradient_descent_outputs(input, weights, trues_multi, alpha, 20)

    numpy_weights, numpy_errors, numpy_weight_history = (gradient_descent_outputs_numpy(input, [.3, .2, .9], trues_multi, alpha, 20))

    print("Numpy final weights: ", numpy_weights)
    print("Final weights match: ", np.allclose(final_weights, numpy_weights))
    print("Error histories match: ", np.allclose(error_history, numpy_errors))
    print("Weight histories match: ", np.allclose(weight_history, numpy_weight_history))


if __name__ == "__main__":
    main()


# Yes all three shrink by the same factor because they are all using the same input and alpha.
# The first output ended closest to its target of 0 at about .0859. No I don't believe this
# means it converged faster, because although it ended closest to its target, they all changed at
# the same ratio so it only ended up closest because it started closest. Output 2 was still fairly
# far off from its target, which means a single sensing doesn't always mean it reaches its target
# quickly, but if you kept doing iterations it would keep reducing the error.