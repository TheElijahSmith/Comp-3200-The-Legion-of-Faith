

# The Trial of Reflection

## Overview

This week we constructed our first fully-fledged neural network with an activation function between the hidden layers. We prove the necessity of the non-linearity that activation functions bring, write a forward pass and backpropagate our errors THROUGH the activation function and then create a full training loop.

## Part Breakdown

### Part 1 - Elijah

In this part we showcase why we need to introduce non-linearity to our model. By creating a simple single layer model and training on a dataset that requires an understanding of non-linear relationships, we prove through the errors that it completely fails to learn anything valuable.

### Part 2 - Robert

This part creates the foundation of the model; a forward pass through the layers with the activation function. We also test run through the untrained network to make sure it works.

### Part 3 - Colson

We implement and show one backprop step here. This is how the model actually learns, and it is built on passing the error (delta) backwards through the model and effectively partitioning the blame backwards by using the derivative of the activation function included in the weight update step.

### Part 4 - Kevin

This is the culmination of the previous steps; the entire model built with a training loop that uses forward passes to get preds, backpropagation to update weights, and a set number of iterations so the model can step through the learning process many times to continually decrease errors and achieve deep learning. We also run through some experimentation to see what hidden layer sizes work consistently for the model.

### Part 5 - Brady

Here we just implement some tests to assert that our previously created functions actually work.