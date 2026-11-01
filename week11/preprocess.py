# Strip trailing newlines. splitlines automatically removes ending newline characters from each line. Eliminating all of the trailing newlines
reviews = open("week11/data/reviews.txt").read().splitlines()
labels = open("week11/data/labels.txt").read().splitlines()

# the following splits reviews into individual words and removes all duplicates within a review
tokens = list(map(lambda s: set(s.split(" ")), reviews))

vocab = set()
for sent in tokens:
    for word in sent:
        if len(word) > 0: vocab.add(word)
vocab = sorted(vocab) # sorted: a set's order changes every run

word2index = {w: i for i, w in enumerate(vocab)} # maps each word to indices
input_dataset = [list({word2index[w] for w in sent if w in word2index})for sent in tokens] # turns reviews into lists of word indices without the duplicates

# Set 1 if review is positive and 0 if negative
labels_to_binary = [1 if label == "positive" else 0 for label in labels]

# Tests to see if the code actually worked
print("Reviews:", len(reviews)) # Verifies that we have all reviews: Expected Result = 25000
print("Labels:", len(labels)) # Verifies we have all labels: Expected Result = 25000
print("Vocabulary Size:", len(vocab)) # Verifies we have acquired every word in the reviews file. Epected Result = 74073 according to lecture

print("First Review Indices:", input_dataset[0]) # displays the first review by the indices
print("First Label:", labels_to_binary[0]) # displays the first review as of the labels_to_binary value.