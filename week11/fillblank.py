import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fillblank(input_dataset, vocab, word2index, epochs=2, alpha=0.05, hidden_size=50, window=2, neg=5):

    # 1. Concatenate all review indices into a 1D array for negative sampling
    concatenated = np.array([idx for review in input_dataset for idx in review])

    # 2. Set seeds and shuffle dataset ONCE per specification
    np.random.seed(1)
    random.seed(1)
    random.shuffle(input_dataset)

    # Initialize weights
    weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
    weights_1_2 = np.zeros((len(vocab), hidden_size))

    # Targets: 1 positive target (index 0), neg negative targets 
    layer_2_target = np.zeros(neg + 1)
    layer_2_target[0] = 1.0

    # 3. Training loop over epochs
    for epoch in range(epochs):
        for review in input_dataset:
            for t in range(len(review)):
                # Sample positive target + 5 random negatives
                target_samples = [review[t]] + list(
                    concatenated[(np.random.rand(neg) * len(concatenated)).astype(int)]
                )

                # Extract context indices (left and right window)
                left = review[max(0, t - window): t]
                right = review[t + 1: min(len(review), t + window)]
                context = left + right

                if len(context) == 0:
                    continue

                # Forward pass
                layer_1 = np.mean(weights_0_1[context], axis=0)
                layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))

                # Backward pass
                layer_2_delta = layer_2 - layer_2_target
                layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

                # Sparse weight updates
                weights_0_1[context] -= layer_1_delta * alpha
                weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha

    # 4. Save trained weights to file
    np.save("weights_0_1_fillblank.npy", weights_0_1)

    return weights_0_1, weights_1_2




if __name__ == "__main__":
    from preprocess import input_dataset, vocab, word2index

    print("Starting training...")
    train_fillblank(input_dataset, vocab, word2index)
    print("Training complete!")