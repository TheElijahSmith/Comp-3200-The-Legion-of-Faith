## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

*Not just "subset" — what makes each layer different from the one above it?*

<!-- Machine Learning is the 2 layer of all the process, is the layer in which you would not give 
the rules, you would give the data so the machine can start learning the patterns and make decision,
since Deep Learning is the 3 layer down that use neural netwoks like a brain, so imagine
that  machine learning are the general tools on a box, and deep learning is a specific complex tool. -->

### 2. A problem traditional programming can't touch

*Original example, and why machine learning suits it.*

<!-- The tradittional programming can help people to detect scams. Machine Learning can solve this problem by feeding a vast historal transactions logs paired with know outcomes into a model to let discover patterns. The model learns complex behavioral profiles, without a human engineer having to write specific rules for them. As also adapts to know scam tactis, data and flag anomalies. -->

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- **Supervised or unsupervised?**
<!-- Unsupervised, because the app organized the songs by a hidden pattern in sound or listening habits instead of a "correct answer key" provided ahead of time. -->
- **Parametric or nonparametric?**
<!-- Parametric, because the algorithm uses a set of adjustable parameters to learn, that summarizes the data and don't keep all raw songs in memory to built the list. -->

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- **Supervised or unsupervised?**
<!-- Supervised, because the model learns with x-rays that a human tag like sick or healty, so knows what to look for. -->
- **Parametric or nonparametric?**
<!-- Parametric, because the neural network uses a number of internal weights to store what it learns, so the size never change no matter how many x-ray you add. -->

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- **Supervised or unsupervised?**
<!-- Supervised, because uses recod purchases like the correct way to predict what a new user is likely to buy. -->
- **Parametric or nonparametric?**
<!-- Nonparametric, because not lear to fix rules ahead of time, just saved the past purchase data and compare to the new users against the saved data. -->

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario:**
<!-- Grouping retail shoppers into 5 distinct groups using raw transactions history saving only center point of each group. -->
- **Supervised or unsupervised?**
<!-- Unsupervised, because there is no pre labels that tells the system who belong to what group, it have to discover the grouping on its own. -->
- **Parametric or nonparametric?**
<!-- Parametric, because the model compresses everything into 5 fixed center points, so the size don't change even if the costumer base grows. -->

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob —**
<!--The tempature setting, this on machine learning represents model weight that you adjust to change how the system behaves.-->
- **(b) the prediction —**
<!--The tempature that produce the AC, this means that is the output the system delivers base on where is the knob set. -->
- **(c) the error —**
<!--The gap between the target tempature and the actual room tempature, tells how far the prediction is from the reality.-->
- **(d) the learning step —**
<!--The act of twisting the dial up or down based on the weather,  is the adjustment you make to shrink the error gap. -->

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` —**
<!--Doubles to 1.7 because is a linear multiplication, scaling the weight by a factor directly scales the output by that exact same factor. -->
- **(b) weight set to `0` —** *(and what that means conceptually)*
<!--Prediction becomes 0.0. This means the input feature is completely ignored by the neuron, getting out of any influence over the final prediction. -->
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)*
<!--Prediction becomes -0.85. Represents an inverse relationship, where an increased in the input active value decreases the output prediction. -->
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)*
<!--Prediction 0.85 was too low. Because subtracting the prediction from the target(1.7 - 0.85 = +0.85) gives a postive error, that means the model estimate undershot the true answer. -->

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

<!-- Finding the right knob is hard because the models hava million of knobs that all are connect to each other at the same time. If you turn off one knob to fix a mistake you could accidentally ruins predictions for thousands of others. We can't calculate the perfect setting with a easy math equation because the data is too high to solve at once. Instead, we have many tiny, trail error tweaks to every knob until the total errors stop getting smaller.-->

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

<!-- Raw audio files of musical compositions along with their corresponding digital score files. My predictions are that the model received a raw audio wave and predicts the exact musical notes, timing and instrument assignments, transcribing live record music into a clean separated sheet music.  -->

### 2. Supervised or unsupervised, and why

<!--The model requires labeled training pairs of raw audio recordings directly matched with human verified files. It needs these target answers during training to learn the precise mathematical mapping between audio frequencies and specific musical notes. -->

### 3. What "success" would look like

*How would you know it was working?*

<!--Success would mean the model achieves high note level precision and recall on complex audio. You would know it was working if a musician could play a live, unscripted session into a microphone, and the software immediately generated an accurate sheet music score with minimal manual correction needed. -->

### 4. What could go wrong

*At least one way this could fail or do harm if deployed carelessly.*

<!--The model could fail on unconventional playing techniques or rare acoustic instruments that weren't well represented in the training data. If deployed commercially in copyright monitoring tools, incorrect note matching could falsely flag original song compositions as copyright violations, penalizing independent musicians. -->

---
