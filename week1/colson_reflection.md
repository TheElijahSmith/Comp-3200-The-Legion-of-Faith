# Foundations Reflection


## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

- AI is the most broad scope that encompasses anything and everything that allows computers to "use human intelligence". This would include things such as reasoning, learning, problem solving, etc.
- Machine learning is a part of AI that uses algorithms to allow computers to learn. Makes decisions and predictions based on the patterns it learned from its data, without needing explicit programming from humans.
- Deep Learning uses neural networks to learn complex patterns from raw data that allows it to excel in image and speech recognition, language processing, etc.

### 2. A problem traditional programming can't touch

- One problem that would be well suited for machine learning would be cancer cell identification. Traditional programming is far worse than machine learning when it comes to dealing with raw data such as images. A machine learning algorithm would process thousands of images either either labeled as containing or not containing cancer and it would learn to identify whether an image does or does not contain cancer on its own.

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- **Supervised or unsupervised?**
Unsupervised
- **Parametric or nonparametric?**
Parametric

- This would be unsupervised parametric because there are no labels already classifying a song as belonging in one playlist or another, but it does have parameters that it adjusts to be able to more accurately classify where it thinks a song should fall.

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- **Supervised or unsupervised?**
Supervised
- **Parametric or nonparametric?**
Parametric

- This is supervised parametric because the images already contain labels for the algorithm to learn from and it has fixed weights that it adjusts in its training.

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- **Supervised or unsupervised?**
Supervised
- **Parametric or nonparametric?**
Nonparametric

- This is supervised nonparametric because it is using previous customers' purchases as labels but it is nonparametric because it is not using fixed labels it is just storing its training examples and comparing it to those.

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario:**
- **Supervised or unsupervised?**
Unsupervised
- **Parametric or nonparametric?**
Nonparametric

- Giving a model thousands of animal images and simply seeing how it will classify them together would be both unsupervised and nonparametric because it has no labels to learn from and no parameters to use to help categorize the images.

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob —**
    - The model's weights
- **(b) the prediction —**
    - The current temp the thermostat thinks it is at
- **(c) the error —**
    - The difference between the current temp and the temp you want it to be at
- **(d) the learning step —**
    - Adjusting the knob a certain direction based on the error

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` —**
    - The prediction will double to 1.7
- **(b) weight set to `0` —** *(and what that means conceptually)*
    - The prediction will be 0 which means that the input has no effect on the prediction and is irrelevant
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)*
    - It would make the prediction -.85 and can be used if the model is signifiying an inverse relationship.
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)*
    - Our current weight is too low because if it had a more significant weight it would have made the prediction higher and more accurate.

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

Because the difficult part is finding which knobs are more significant than others and how they all play a part individually and part of a whole. The more knobs we add the more complex their relationships get and the harder it is to learn each of them

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

I would love to use deep learning to help a robot autonomously navigate different areas. I would use different sensors and cameras and Lidar as my raw input and let my model take the data and determine which areas would be safe to travel.

### 2. Supervised or unsupervised, and why

I think this could be supervised because I would provide the model with labeled obstacles and allow it to navigate the environment around it with those training labels. Though I think some of it would be unsupervised if it encounters an obstacle that was not labeled in training.

### 3. What "success" would look like

Success would look like the robot being able to navigate any obstacles I put in its way and accurately make it through a room without crashing into anything.

### 4. What could go wrong

The main way I see this going wrong is with obstacles that I don't label in training. With a good enough model I believe this wouldnt be an issue but it very well could be a prominent issue I would need to overcome.

---

## Before you open the PR

- [ ] All four parts answered — check against the headings above, not your memory
- [ ] Part 2: both axes **and** a justification, for all four scenarios
- [ ] Part 2 #4 uses a combination you didn't already use
- [ ] Part 4 #4 is answered — it's the one people skip
- [ ] Your own words throughout (see below)
- [ ] The file is `week1/yourname_reflection.md` — not this template

Week 1 branches straight off `main`; there's no week branch until Week 2
(`CONTRIBUTING.md` §1).

```bash
git checkout main
git pull origin main
git checkout -b week1/alia-reflection        # your name, not Alia's
git add week1/alia_reflection.md
git commit -m "[Week1] Add reflection for Alia Mehta"
git push -u origin week1/alia-reflection
```

Open the PR into `main`, ask a teammate to review it — they're confirming it's
complete and thoughtful, they are not grading it — and merge before Sunday 11:59 PM.

---

## On AI tools

You're welcome to use them to check your understanding. Every answer has to be in your
own words and reflect your own thinking. If I can paste your answer into a search box
and find it verbatim, that's a problem.

**And say so when you use them.** A line in your PR description naming what you used
and what you used it for is enough — the pull request template asks for it directly.
Acknowledged use is ordinary professional practice. Unacknowledged use is an academic
integrity issue.

Don't let an assistant write your reflection. Write it. *Then* ask it where you've been
unclear — and rewrite.
