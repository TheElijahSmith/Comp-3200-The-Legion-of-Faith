# Visions

## Overview

This week we upgrade our model from last week by adding the ability of recognizing word order into our model. By doing this, we create an RNN model
which recursively updates the value of a sentence after reading each word to reach the final prediction.


## Part Breakdown

### Part 1 - Robert

Here we preprocess out data by creating a vocabulary dictionary and sorting it, as well as tokenizing all of the sentences in the data.

### Part 2 - Kevin

Here we implement an RNN forward pass.

### Part 3 - Brady

Here we is the backpropagation through time which is the heart of the model's training by combining 2 separate deltas during backprop.

### Part 4 - Elijah

This is the actual full training loop for the model on the data which also produces a log of the perplexities over time and the final trained weights.

### Part 5 - Colson

This shows a comparison between the model actually predicting on some test sentences and reality. It showcases the power of our model and its shortcomings.

### Part 6 - Kevin

Tests to prove that our functions work.

### Part 7 - Elijah

An explanation of the Holocron's warning for this week and what it means for the model to learn structure vs understanding.

