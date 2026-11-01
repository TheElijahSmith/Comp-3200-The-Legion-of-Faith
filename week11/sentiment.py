import numpy as np
import pickle
from preprocess import reviews, labels, vocab, word2index

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

np.random.seed(1)
alpha = 0.04
iterations = 2
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

def train(x, y):
  #Forward pass on calculating the hidden lyaer
  layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
  #forward pass on calculating the prediction
  layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
  #Backward pass calculating the output error
  layer_2_delta = layer_2 - y
  #backwardpass on sending the error to the hidden layer
  layer_1_delta = layer_2_delta.dot(weights_1_2.T)
  layer_1_delta *= layer_1 * (1 - layer_1)
  #Updates the weights
  weights_0_1[x] -= layer_1_delta * alpha
  weights_1_2[:] -= np.outer(layer_1, layer_2_delta) * alpha
  
  return layer_2

#Code to train the network
for iteration in range(iterations):
  correct = 0
  for i in range(24000):
    x = reviews[i]
    y = labels[i]
    
    prediction = train(x, y)
    
    if (prediction[0] >= 0.5) == (y == 1):
      correct += 1
      
      if (i + 1) % 1000 == 0:
        accuracy = correct / 1000
        print(
          f"Iteration {iteration + 1}, "
          f"Review {i + 1}, "
          f"Accuracy: {accuracy:.2%}"
        )
        correct = 0

def predict(x):
  layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
  
  layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
  
  return layer_2

# Code to evaluate the network on the test set
correct = 0
for i in range(24000, 25000):
  x = reviews[i]
  y = labels[i]
  prediction = predict(x)
  if (prediction[0] >= 0.5) == (y == 1):
    correct += 1
    
test_accuracy = correct / 1000
print(f"Test accuracy: {test_accuracy:.2%}")

with open("sentiment_data.pkl", "wb") as file:
  pickle.dump({
    "weights_0_1": weights_0_1,
    "word2index": word2index,
    "vocab": vocab
  }, file)









