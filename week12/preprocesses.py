with open("week12/data/qa1_train.txt") as f:
    lines = f.readlines()[:1010] # open file and read first 1010 lines


tokens = []

for line in lines:
    words = line.lower().strip().split(' ') # lowercase, strip trailing newlines, and split on whitespace
    words = words[1:] # remove line number
    tokens.append(words)




# same setup as week 11
vocab = set()

for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = sorted(vocab)




# same as week 11
word2index = {word: i for i, word in enumerate(vocab)}


# The following function takes the sentences from the data folder and turns each word into their respective indice from the set above.
def words2indices(sentence):
    return [word2index[word] for word in sentence]



if __name__ == "__main__":
    print(len(lines))

    # test. If this works, we should see line 1 of qa1_train.text written out in lower case
    print(tokens[0])

    # test. Should get 82 and the full list. *** Note made after tests, read the comments at the BOTTOM of the file
    print("Vocabulary Size:", len(vocab))
    print(sorted(vocab))

    # test. Should output the number corresponding to where mary is in the set
    print(word2index["mary"])

    # test. Should turn line 1 in the data folder to its respective indices
    print(words2indices(tokens[0]))