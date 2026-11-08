import numpy as np

from preprocesses import tokens, words2indices, vocab
from rnn_forward import predict


weights = np.load("week12/trained_weights.npz")

embed = weights["embed"]
recurrent = weights["recurrent"]
start = weights["start"]
decoder = weights["decoder"]

# select three statements
test_indices = [1000, 1003, 1006]

for i in test_indices:
    sentence = tokens[i][1:]
    sent = words2indices(sentence)

    print(f"Sentence {i}: {sentence}")
    # run the sentence through the trained RNN
    layers, loss = predict(sent, embed, recurrent, start, decoder)

    
    print(f"{'Context':<25} {'Actual':<15} {'Predicted':<15}")
    print("-" * 55)

    for layer_idx in range(2, len(layers)):
        prediction = layers[layer_idx]["pred"]
        # get the word with the highest predicted probability
        predicted_idx = np.argmax(prediction)
        predicted_word = vocab[predicted_idx]

        # get words already seen and the actual next word
        context = " ".join(sentence[:layer_idx - 1])
        actual_word = sentence[layer_idx - 1]

        print(f"{context:<25} {actual_word:<15} {predicted_word:<15}")
