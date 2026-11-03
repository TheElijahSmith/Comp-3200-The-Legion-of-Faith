from preprocess import word2index, input_dataset, labels_to_binary, vocab
from sentiment import weights_0_1, train, predict
import numpy as np
from similarity import similar
import random
from fillblank import sigmoid

# Add layer_2 and layer_2 target to return statement of fillblank for testing
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


    return weights_0_1, weights_1_2, layer_2, layer_2_target





def test_preprocessing():
    # Make sure vocab size is the correct length
    assert len(word2index) < 80000 and len(word2index) > 70000, "Vocab size is wrong."

    # Check that the index for the is an int and between 0 and the vocab length
    assert (word2index['the'] >= 0 and word2index['the'] < len(word2index) and
            type(word2index['the']) is int), " 'the' does not have a valid index"

def test_row_lookup():

    # Example indices for a sentence
    x = [2, 5, 9]

    # Sum along axis_0
    axis_sum = weights_0_1[x].sum(axis=0)

    # Individually sum weights elementwise
    ele_sum = weights_0_1[2] + weights_0_1[5] + weights_0_1[9]

    # Assert that those are the same
    assert np.array_equal(axis_sum, ele_sum), "Sum along axis 0 != to sum of individiual rows"

def test_sentiment():
    # Grab small input set from reviews, just the first 350
    for i in range(350):
        x = input_dataset[i]
        y = labels_to_binary[i]
        train(x, y)

    correct = 0

    #test over 1000 samples from the dataset and report accuracy
    for i in range(350, 1350):

        x = input_dataset[i]
        y = labels_to_binary[i]
        prediction = predict(x)
        if (prediction[0] >= 0.5) == (y == 1):
            correct += 1
            
    test_accuracy = correct / 1000
    assert test_accuracy > .6, "Sentiment train not training above random chance"


def test_similarity():
    # some random test words
    words = ['great', 'bad', 'the', 'awful', 'scary', 'a', 'movie']

    # check that similar returns the same word the first because a dist of 0 means its the closest.
    for word in words:
        assert similar(word)[0] == word,"similar not returning the same word first"

def test_neg_sampling():
    #test with 7 negatives
    neg = 7

    #grab just 50 reviews to train on so it doesn't take forever
    _, _, layer_2, layer_2_target = train_fillblank(input_dataset[:50], vocab, word2index, neg = neg)

    assert layer_2.shape == (neg+1,), "layer2 shape is incorrect"

    # layer_2_target should have a 1 and the number of negatives 0s
    assert np.array_equal(layer_2_target,[1, 0, 0, 0, 0, 0, 0, 0]), "Layer 2 target is not correct"

if __name__ == '__main__':
    tests = [name for name in dir() if name.startswith('test_')]
    passed = []
    failed = []
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            passed.append(test_name)
        except AssertionError as e:
            failed.append(test_name)
    for name in passed:
        print(f'PASS: {name}')
    for name in failed:
        print(f'FAIL: {name}')
  
