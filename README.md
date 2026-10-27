# Happy or Not? — Teach a Neural Network to Read Faces

**Live (try it in the browser):** https://professor-jeff.github.io/happy-face-nn/

Third in the COMP 138 series after [pick-activity](https://github.com/Professor-Jeff/pick-activity)
and [tictactoe-nn](https://github.com/Professor-Jeff/tictactoe-nn). The same small network,
written from scratch in plain JavaScript, now looks at pictures: 10×10 black-and-white faces.
A second model, convolutional, sits behind a switch so the two can be compared on identical data.

This time there are no rules to copy and no perfect player to imitate. **The student makes the
training data.** They scroll a gallery of generated faces, pick the ones they consider happy, and
the network learns *their* definition. Then it sits an exam on faces it has never seen.

## What the demo shows

1. **Pick** — a gallery of 64 faces (happy, sad and neutral mouths; varied eyes, head outlines,
   noses, tilts). Every face the student picks is labeled *happy*; faces they skip are *not happy*.
2. **Split, then train** — the picks plus about twice as many unpicked faces become the training
   set. Unpicked happy faces, the remaining others, and 12 junk images (noise, stripes, scrambled
   faces, blobs) are held out for the test set. 300 epochs of gradient descent, with the loss curve,
   the test accuracy curve, and a dozen training faces updating live.
3. **The exam** — every held-out face with the network's probability, the true label, and a
   right/wrong color. Junk is flagged separately and not counted in the accuracy.
4. **Draw a face** — a 10×10 grid to paint on, with a live prediction. Drawing is for testing only.

**Two models, one switch.** Step 2 has a *Fully connected | Convolutional* switch. Both train on the
same picks and sit the same exam.

- *Fully connected*: 100 pixels → 8 relu hidden → 1 sigmoid output, 817 weights. It has no idea that
  the same shape in a different place is the same thing, so a smile drawn two rows too high fails.
- *Convolutional*: six 3×3 filters slid across the picture (8×8 feature maps, relu), then the
  strongest response in the left half and the right half of each map (12 numbers), then the same
  8 hidden → 1 tail. 173 weights. Because the pooling forgets how high up a shape was, the
  too-high smile usually passes. **Show the network** draws the filters and their feature maps live.

**Shifted copies (data augmentation).** A spinner in step 2 (0 to 3, default 0) also trains on
every training face shifted by up to that many cells in every direction. Copies that would push
part of a face off the 10×10 edge are dropped: a smile with its mouth cut off is a mislabeled
example, and those hurt more than the extra copies help. 3 is as far as a face can move before it
hits the edge, so the range stops there. The live count shows how many faces the copies add.

Measured over 12 random galleries with 5–8 picks each (unseen accuracy / unseen happy faces caught /
unseen happy faces shifted up two rows caught), 300 epochs:

| model | radius 0 | radius 1 | radius 2 | radius 3 |
|---|---|---|---|---|
| fully connected | 84 / 67 / 27 % | 73 / 58 / 18 % | 73 / 57 / 40 % | 73 / 58 / 49 % |
| convolutional | 79 / 58 / 61 % | 82 / 68 / 75 % | 83 / 68 / 73 % | 83 / 75 / 82 % |

The fully connected model never gets past about half of shifted faces and always pays in exam
accuracy; the convolutional model gains on every column.

## Where the math lives

`index.html` is one self-contained file. The neuron math is in the `<script>` block under
**PART 2: the network**:

- `forward` — 100 pixels × W1 + b1, relu, × W2 + b2, sigmoid → probability of “happy”.
- `trainEpoch` — binary cross-entropy loss, then backpropagation. The same line as the other demos:

```js
const dz2 = (prob - y) / n;      // "predicted minus correct"
```

**PART 2b** is the convolutional model: `convForward` slides the filters and max-pools, and
`convTrainEpoch` backpropagates through the pooling (the gradient goes to the spot that won the max)
and the convolution (each filter's gradient adds up every place it was applied). `MODELS` is the
registry the page switches between.

Both: full-batch gradient descent, learning rate 0.5, 300 epochs. Sizes are deliberately fixed so
the lesson is about data and wiring, not knobs.

**PART 1: the faces** is the generator (seeded, so a classroom run is repeatable) and
`splitData`, which builds the training and test sets from the student's picks.

## Deploying

Edit `index.html` and push to `main`. GitHub Pages redeploys within about a minute.
