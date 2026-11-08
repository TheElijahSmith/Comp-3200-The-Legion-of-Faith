import numpy as np
from rnn_forward import predict, compute_perplexity

def train_step(sent, embed, recurrent, start, decoder, one_hot, alpha):
  # This will run the forward pass
  layers, total_loss = predict(
    sent, embed, recurrent, start, decoder
  )

  # Walk backward through the layers
  for layer_idx in reversed(range(1, len(layers))):
    
    target = sent[layer_idx - 1]
    
    layer = layers[layer_idx]
    layer['output_delta'] = layer['pred'] - one_hot[target]
    
    new_hidden_delta = layer['output_delta'].dot(decoder.T)
    
    if layer_idx == len(layers) - 1:
      layer['hidden_delta'] = new_hidden_delta
    else:
      layer['hidden_delta'] = (
        new_hidden_delta
        + layers[layer_idx + 1]['hidden_delta'].dot(recurrent.T)
      )
  # calculate the hidden delta for the starting layer
  layers[0]['hidden_delta'] = (
    layers[1]['hidden_delta'].dot(recurrent.T)
  )
  # calculate the learning rate
  step_size = alpha / len(sent)
  
  # update the starting hidden state
  start -= layers[0]['hidden_delta'] * step_size # here
  
  for layer_idx, layer in enumerate(layers[1:]):
    
    decoder -= np.outer(
      layer['hidden'], # here
      layer['output_delta']
    ) * step_size
    
    if layer_idx + 2 < len(layers):
      # get the hidden gradient for the word
      grad_h = layers[layer_idx + 2]['hidden_delta']
      # update the word embedding weights
      embed[sent[layer_idx]] -= grad_h * step_size
      # update the recurrent weights
      recurrent -= np.outer(
        layers[layer_idx]['hidden'],
        grad_h
      ) * step_size
  perplexity = compute_perplexity(total_loss, len(sent))
  return perplexity


