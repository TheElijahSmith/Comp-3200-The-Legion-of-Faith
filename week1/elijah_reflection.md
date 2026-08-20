
*In your copy, delete the block above and the italic hints as you go.*

---

## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

*Not just "subset" — what makes each layer different from the one above it?*

Artificial intelligence is essentially just whenever a machine provides an answer to a problem 
following some set of orders. In machine learning, the computer is the one who comes up with
the rules rather than the programmer. In deep learning, the computer utilizes neural networks to
come up with those rules.

### 2. A problem traditional programming can't touch

*Original example, and why machine learning suits it.*

If you have a set of complex data and can't really figure out a good way to predict how a new data
point would behave, you can't program any rules for the machine to follow because you don't know them
yourself. With machine learning, you can have the machine come up with the rules itself and now you can
make real predictions on the dataset.

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- Unsupervised - it is grouping unlabeled datapoints together.
- Parametric - it is adjusting a set number of parameters.

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- Supervised - the data is labeled and is making preds and comparing them to truth.
- Parametric - there is a set number of parameters that are being adjusted.

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- Unsupervised - It's grouping data points together without labels.
- Nonparametric - It's "remembering" previous entries and didn't have any set parameters.

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- A photo website has thousands of labeled images of people and animals and it remembers what those types of images
  look like and generates new images using learned parameters and makes new parameters as it goes.
- Supervised - the data is labeled.
- Nonparametric - it remmebers the data and makes new parameters as it learns.

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- (a) the knob — you turn the knob to hot and cold like a weight parameter that adjust the importance of inputs.
- (b) the prediction — is the turning of the knob to where you think it should go to reach your desired temp. You're not
    sure if you're right, but it's an educated guess.
- (c) the error — overturning or underturning the knob. Basically any difference you are from what you're desired temp was.
- (d) the learning step — how big of a turn you are making to try to adjust in the direction of the ideal temp.

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- (a) weight doubled to `0.2` — It becomes 1.7
- (b) weight set to `0` — The pred becomes 0. Infact all preds woud become zero, 
    meaning that this input would never have an effect on the final pred, ever.
- (c) weight negative, `-0.1` — The pred is -.85, meaning this input would have an
    opposite correlation to the postive pred class.
- (d) if the answer should have been `1.7` — It's too low. If the pred is .85 then it is
less than 1.7 so we need to bump up our weight so we can move our pred in the direction towards
the correct number

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

Because in many cases there are thousands or millions of knobs and we just don't know how
every single little piece of information needs to be interpreted. To calculate them perfectly,
we would have to have a perfect understanding of what it is we are calculating, and as we are in
the real world where nothing is ever perfect, we aim for as close as we can get, and so we use DL
to fine tune way faster than a human could.

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

I want to figure out how to take a robotic arm and enable it to intelligently perform actions
on its own.
The input data would be images from a camera and sensor readings indicating touch and
maybe audio allowing for voice commands.
(think of the free standing robotic arms that Tony Stark has in Iron Man)

### 2. Supervised or unsupervised, and why

Supervised. The AI would be trained on supervised data to understand what object it was looking 
at or how far it was away and same with the audio..
The sensor readings would also be labeled (i.e. "touching" or "not touching") and it would 
work in tandem with the camera and audio.

As for the physical arm movements, those would be semi-supervised, with the AI having to correctly
understand what successful movements were following guidelines I established.

### 3. What "success" would look like

*How would you know it was working?*

If the robot could do simple commands like, "pick up that magnet and put it in that slot" and it refrains
from jerky movements which could cause catastrophic failures.

### 4. What could go wrong

*At least one way this could fail or do harm if deployed carelessly.*

If someone tried to set up this model with improper training and recognition, it could perform dangerous,
sporadic movements as well as forcefully break things apart or even misunderstand what object was intended
and attempt to grab a person or something else.

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
