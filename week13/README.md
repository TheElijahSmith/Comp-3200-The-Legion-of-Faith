# Visions

## Overview

This week we created a framework made of tensors built for creating neural networks.
It allows the user to simply design the forward pass and it can auto back-prop for you.
We have standard linear layers as well as some activation functions as options for using the framework.


## Part Breakdown

### Part 1 - Elijah

This is the bones of the framework: a tensor class which acts as a carrier for data, its creators,
id, and the operation that created it. It has functions built in to backprop when given gradients from
its children matrices. Only matrix addition is programmed for the tensor in this part.

### Part 2 - Kevin

This adds the rest of the rules for the other matrix operations of negation, subtraction, element multiplying,
matrix multiplying, transposing, summing over a dimension, and expanding along a dimension.

### Part 3 - Robert

Here we create the optimizer class which acts as the adjustment of all the parameters in the model as well as
a base layer class which holds parameters, a linear class which holds a weight and bias, and sequential class
which steps forward through a network of layers.

### Part 4 - Brady

Here we train a simple two layer linear model and prove that our code works.

### Part 5 - Colson

We implement activation function versions of our layer class which allows our model to embrace nonlinearity
and our framework to be much stronger.

### Part 6 - Kevin

Tests to prove that our functions work.

### Part 7 - Elijah

An explanation of "removing the warning" and how that completely invalidates a model's ability without it being expressly obvious.

