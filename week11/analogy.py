import numpy as np
import math
from collections import Counter
from preprocess import word2index

weights_0_1 = np.load("weights_0_1_fillblank.npy")

def analogy(positive, negative, k=5):
    q = np.zeros(weights_0_1.shape[1])

    # adds embedding vectors of positive words
    for w in positive:
        q += weights_0_1[word2index[w]]

    # adds embedding vectors of negative words
    for w in negative:
        q -= weights_0_1[word2index[w]]

    scores = Counter()

    # Euclidean distance between q and each words embedding
    for word, idx in word2index.items():
        d = weights_0_1[idx] - q
        scores[word] = -math.sqrt(np.sum(d * d))

    # exclude the input words from the result
    return [w for w, _ in scores.most_common(20)
            if w not in positive + negative][:k]



if __name__ == "__main__":
    print("Required analogy:")
    print(analogy(['terrible', 'good'], ['bad']))

    print("Custom analogy:")
    print(analogy(['great', 'bad'], ['good']))

    print("\nPart 4 Similarity Results:")

    index2word = {idx: word for word, idx in word2index.items()}

    for target in ['terrible', 'great', 'bad']:
        target_vec = weights_0_1[word2index[target]]
        distances = np.linalg.norm(weights_0_1 - target_vec, axis=1)
        nearest = np.argsort(distances)[1:11]

        print(target, [index2word[i] for i in nearest])



#                   For the notebook
# When comparing the two embeddings, I think the sentiment classifier gave better results overall than the 
# fill-in-the-blank model. For example, when we looked at words similar to "terrible" in the sentiment model, 
# we got words like "disappointment," "boring," "annoying," and "dull," which all make sense because they have 
# negative meanings. However, the fill-in-the-blank model gave us words like "success," "values," "streets," 
# and "giant," which don't really relate to "terrible." We saw something similar with "bad," where the sentiment 
# model returned words like "horrible," "disappointed," and "lame," while the fill-in-the-blank model gave us 
# words like "extra," "pops," and "chapter." The fill-in-the-blank model did have some good results, like connecting 
# "great" with "exciting" and "delightful," but it was much less consistent. I think the sentiment model performed 
# better because it was specifically trained to recognize positive and negative words, while the fill-in-the-blank 
# model was just learning how words were used around each other. This made the sentiment model better at finding words 
# with similar meanings in our results.