import numpy as np

#Initialize the four weight tensors and one-hot matrix using seed 1.
def init_weights(vocab_size, embed_size=10):
    np.random.seed(1)
    
    embed = (np.random.rand(vocab_size, embed_size) - 0.5) * 0.1
    recurrent = np.eye(embed_size)
    start = np.zeros(embed_size)
    decoder = (np.random.rand(embed_size, vocab_size) - 0.5) * 0.1
    one_hot = np.eye(vocab_size)
    
    return embed, recurrent, start, decoder, one_hot

#Compute numerically stable softmax by subtracting the maximum value.
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def predict(sent, embed, recurrent, start, decoder):
    # Layer 0 holds the initial start state; it has no 'pred' of its own
    layers = [{'hidden': start}]
    total_loss = 0.0
    
    for layer_idx, target_word_idx in enumerate(sent):
        # Predict the next word distribution using the hidden state from the previous layer
        prev_hidden = layers[layer_idx]['hidden']
        pred = softmax(prev_hidden.dot(decoder))
        
        # Accumulate cross-entropy loss for the target word
        loss = -np.log(pred[target_word_idx])
        total_loss += loss
        
        # Compute the new hidden state by combining recurrence and word embedding
        next_hidden = prev_hidden.dot(recurrent) + embed[target_word_idx]
        
        # Append layer dictionary containing hidden state and prediction distribution
        layers.append({
            'hidden': next_hidden,
            'pred': pred
        })
        
    return layers, total_loss

#Compute per-step perplexity: exp(total_loss / len(sent)).
def compute_perplexity(total_loss, sent_len):
    division = total_loss / sent_len
    return np.exp(division)