## Part 1 — The Landscape (4 pts)

### 1. AI vs. machine learning vs. deep learning

Deep Learning is the deepest layer to Artificial Intelligence. It mainly covers neural networks composed of things such as Division Trees and SVMs. Things like Decision Trees and SVMs are things that compose Deep Learning, essentially the middle ground of the multi-layered AI cake. Ai itself is the outer layer and is primary composed of rule based expert systems.

### 2. A problem traditional programming can't touch

You want to explain to a math student the reason as to why specific solving methods had to be used to get their answer. Essentially the Machine is looking to take the solution and work the student made to build the theorems they used.

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

- **Supervised or unsupervised?** - Unsupervised. The music app is being trained on what you are providing it and is grouping it together. Grouping being a huge feature of unsupervised learning.
- **Parametric or nonparametric?** - Nonparametric. This app grows with the number of songs you listen to and doesn't forget them because it has to be able to group them together based on similar songs.

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

- **Supervised or unsupervised?** - Supervised. The hospital is training the model. The already have the answers and are ready to adjust.
- **Parametric or nonparametric?** - Parametric. The hospital is adjusting each knob to resolve errors and the dataset is a fixed count of 100,000.

### 3. Retail site — recommendations from the 10 most similar past customers

- **Supervised or unsupervised?** - Supervised. The dataset is looking at what other people have done so the structure is made from the previous data. The beginning of the model's lifetime however would not be supervised as there is no data or not enough data to go off of.
- **Parametric or nonparametric?** - Nonparametric. The model is remembering the data as it comes in. So the data is always flexible and growing. 

### 4. Your own scenario

- **The scenario:** - A library has made a system that recommends people books based on a genre of their selection and what people of a similar age have checked out from that genre.
- **Supervised or unsupervised?** - Supervised. The system is giving you recomendations based on a preset structure of age and genre.
- **Parametric or nonparametric?** - Nonparametric. The system has to remember not only every person who has checked out a book, but what book and genre that person checked out so it can recommend people of a similar age that same book if they chose the same genre.

---

## Part 3 — The Knobs Mental Model (6 pts)

### 1. The thermostat analogy

- **(a) the knob — the knob we are turning to find the temperature.
- **(b) the prediction — what temperature we are guessing we turned the knob to.
- **(c) the error — if we did not accurately predict the temperature we were looking for.
- **(d) the learning step — adjusting the knob by tuning it in the direction we think will change the temperature to out target.

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` — 1.7
- **(b) weight set to `0` — 0
- **(c) weight negative, `-0.1` — -.85
- **(d) if the answer should have been `1.7` — The weight needed to be increased so it was too low. Our test of 0.2 was what got us to 1.7.

### 3. The big picture

This probem is difficult because our problems are not going to be as simple as a singular knob to turn. The exact positions will also not be easily found because each knob will do different things. There are probably some people who can answer harder probelms with ease but that is an anomaly in the grand scheme of things.

---

## Part 4 — Your Deep Learning Problem (4 pts)

### 1. The problem

I would like to be able to take books and create maps to help people understand the plots of them better. complex stories are facinating because they have a complex plot. But sometimes that can be hard to follow depending on how many subplots there are and how each one progresses.
The imput data would be the book itself with the prediction being what parts of the book lead to subplots resolving.

### 2. Supervised or unsupervised, and why

This is definietely a supervised problem. The data itself is built off of the plot of the book, meaning the model has to fed the book and then deduce how each subplot is progressing and resolving.

### 3. What "success" would look like

Success would look like the model giving you a list of the main plot and subplots of the book and then explaining to you what drives each plot forward and how it is.

### 4. What could go wrong

1. Plots that dont finish in a singular novel require more novels that may not have been in the initial input.
2. The model might find more subplots than there really was. An answer might be to allow the reader to choose a subplot if applicable.