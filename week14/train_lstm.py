
import numpy as np

from tensor import Tensor
from layers import Embedding, LSTMCell, CrossEntropyLoss
from optim import SGD


# Settings
np.random.seed(0)

embed_size = 512
hidden_size = 512

batch_size = 16
bptt = 25

alpha = 0.032
epochs = 1


# Read the training text
with open("data/qa_oldtongue.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Build the vocabulary
vocab = sorted(set(text))

char_to_idx = {char: i for i, char in enumerate(vocab)}
idx_to_char = {i: char for i, char in enumerate(vocab)}

data = np.array([char_to_idx[char] for char in text])

print("Vocabulary size:", len(vocab))
print("Total characters:", len(data))


# Batch the text
usable_length = (len(data) // batch_size) * batch_size
data = data[:usable_length]

data = data.reshape(batch_size, -1)

inputs = data[:, :-1]
targets = data[:, 1:]

sequence_length = inputs.shape[1]


# Build the model
embedding = Embedding(len(vocab), embed_size)

lstm = LSTMCell(embed_size, hidden_size, len(vocab))

loss_function = CrossEntropyLoss()

parameters = [embedding.weight]

for layer in [
    lstm.xf, lstm.xi, lstm.xo, lstm.xc,
    lstm.hf, lstm.hi, lstm.ho, lstm.hc,
    lstm.w_ho
]:
    parameters.append(layer.weight)

    if layer.bias is not None:
        parameters.append(layer.bias)

optimizer = SGD(parameters, alpha)


# Train the LSTM
for epoch in range(epochs):

    hidden = lstm.init_hidden(batch_size)

    total_loss = 0.0
    total_characters = 0
    batch_number = 0

    for start in range(0, sequence_length, bptt):

        end = min(start + bptt, sequence_length)

        # Detach the hidden state every BPTT chunk
        hidden = (
            Tensor(hidden[0].data, autograd=True),
            Tensor(hidden[1].data, autograd=True)
        )

        chunk_loss = None

        for position in range(start, end):

            current_input = inputs[:, position]
            current_target = targets[:, position]

            embedded = embedding.forward(current_input)

            prediction, hidden = lstm.forward(embedded, hidden)

            loss = loss_function.forward(prediction, current_target)

            if chunk_loss is None:
                chunk_loss = loss
            else:
                chunk_loss = chunk_loss + loss

            # Track the running loss
            total_loss += float(loss.data)
            total_characters += 1

        # Average the loss across this chunk
        chunk_steps = end - start
        chunk_loss = chunk_loss * (1.0 / chunk_steps)

        # Backpropagate and update weights
        chunk_loss.backward()
        optimizer.step()

        batch_number += 1

        # Calculate running perplexity
        average_loss = total_loss / total_characters
        perplexity = np.exp(average_loss)

        if batch_number % 10 == 0 or batch_number == 1:
            print(
                "Epoch:", epoch + 1,
                "Batch:", batch_number,
                "Perplexity:", round(perplexity, 2)
            )

    # Decay the learning rate after each epoch
    alpha *= 0.99
    optimizer.alpha = alpha

    print(
        "Finished epoch", epoch + 1,
        "Final perplexity:",
        round(np.exp(total_loss / total_characters), 2)
    )

                             

