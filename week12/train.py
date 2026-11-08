import numpy as np
from preprocesses import words2indices, word2index, tokens
from rnn_forward import compute_perplexity, init_weights
from rnn_bptt import train_step

TRAIN_N = 1000

alpha = .001
iterations = 30000

def train(alpha, iters, TRAIN_N):
    #initialize the weights
    embed, recurrent, start, decoder, one_hot = init_weights(vocab_size=len(word2index))

    perplexity_log = []
    
    # Main loop
    for it in iters:
        # Get the current sentence. it % train_n will cycle through the first TRAIN_N lines
        #  however many times it/TRAIN_N is
        sent = words2indices(tokens[it % TRAIN_N])

        # train the model this step and get perplexity
        perplexity = train_step(sent, embed, recurrent, start, decoder, one_hot, alpha)

        # every 1000 iters, add the current iteration and perplexity to a log of the training
        if (it+1) % 1000 == 0:
            perplexity_log.append((iter, perplexity))
            print((iter, perplexity))

    np.savez("perplexity_log.csv", perplexity_log)