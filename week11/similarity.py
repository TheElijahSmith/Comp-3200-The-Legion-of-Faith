import numpy as np
from sentiment import word2index, weights_0_1
from collections import Counter

# This func returns the k nearest words to the target according
#  to euclidean distance using learned weights.
def similar(target, k=10):

    target_index = word2index[target]

    # Get target weights, they will be at the same index as the
    #  index of the word
    target_vec = weights_0_1[target_index]

    # init the euclidean scores as a counter
    scores = Counter()

    # Loop through all the words in word2index
    for word, index in word2index.items():
        #don't return the same word in closest
        if index != target_index:
            # Find the difference between the word's weight and the target
            diff = weights_0_1[index] - target_vec

            # Take the square root of the sum of the axis distances squared
            #  Make it negative so the smallest distances have the highest score
            scores[word] = -np.sqrt(np.sum(diff*diff))
    #returns the 10 highest scored words
    closest_words = []
    for word, val in scores.most_common(k):
        closest_words.append(word)
    return closest_words

print(similar('terrible'))
print(similar('great'))
print(similar('bad'))

"""
For notebook:
Closest to 'terrible':
    ['disappointment', 'boring', 'annoying', 'dull', 'avoid', 'poor', 
    'save', 'poorly', 'disappointing', 'laughable']
Closest to 'great':
    ['solid', 'enjoyed', 'charlie', 'subtle', 'gem', 'best', 'quiet', 
    'masterpiece', 'definitely', 'hilarious']
Closest to 'bad':
    ['disappointed', 'insult', 'horrible', 'terrible', 'lame', 'turkey', 
    'poor', 'ridiculous', 'pointless', 'painful']
"""

    