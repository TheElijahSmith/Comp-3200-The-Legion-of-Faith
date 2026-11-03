The sentiment model and the fill-in-the-blank model created different maps because they were trained to accomplish different goals. The sentiment model
was trained to recognize whether reviews were positive or negative. This would result while training it would notice where to place words that have similar sentiment near each other.
An example would be "terrible" it would return "disappointment", "boring", "annoying", and "dull" All the words here would be commonly associated with the word "terrible".
The Fill in the blank model was trained to predict missing words based on surrounding context instead of identifying sentiment. This would mean "terrible" would be linked with
"success", "values", "streets", and "giant", which were less obviously related in meaning.
Korrith's warning is an important engineering principle because ebedding does not understand what a word means, it will just predict what words are going to be next to it majority of
the time. This means embeddings needs to be feed a lot of data, but also data that will output the desired results. If the data is bad it will output biases and bad results 
depending on what the embedding is used for. This is why teams building embeddings need to be careful with what data they give to the embeddings.
