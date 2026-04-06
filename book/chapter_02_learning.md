# Chapter 2: How a Model Learns

> *"Training a model is like raising a child: you give feedback, hope it improves,
> and eventually realize you misconfigured the learning rate from the start."*

## 2.1 The fundamental question: how does the model know it is wrong?

You have a model with millions of parameters. You feed it a grayscale image and
it produces something colorized. But how do you know if the colorization is good
or bad?

You need a **loss function**: a mathematical formula that turns "how wrong is the
model" into a single number.

If the model is perfect, loss = 0.
If the model is bad, loss is large.

The goal of training: **minimize the loss**.

## 2.2 The loss function

The simplest loss: **Mean Absolute Error (L1)**

```
L1 = (1/N) * sum( |prediction - ground_truth| )
```

The model predicts that pixel (100, 100) has color (a=0.3, b=-0.1).
The ground truth is (a=0.5, b=0.2). The error is:

```
|0.3 - 0.5| + |-0.1 - 0.2| = 0.2 + 0.3 = 0.5
```

Do this for every pixel in every image in a batch, take the average, and you
get the loss. One number summarizing "how bad you are right now".

For colorization, L1 has a known problem: if grass could be light green,
medium green, or dark green, the model predicts the *average* of all plausible
colors. A dull, desaturated green. Technically correct, perceptually boring.
That is why we use more sophisticated loss functions (Chapter 8).

## 2.3 The gradient

![Gradient descent](images/gradient_descent.png)

*Left: a ball rolling down the loss curve toward the minimum, one gradient step
at a time. Right: the gradient is the slope of the tangent at any point. It tells
you which direction makes the loss go up, so you go the other way.*

Now that we have a number (the loss), how do we reduce it?

Imagine you are on a foggy mountain and want to reach the valley (loss = 0).
You cannot see far, only what is directly around you. The best strategy: feel
the slope and step in the steepest downward direction.

The **gradient** is exactly that: for each parameter in the model, it says
*how much and in which direction* that parameter needs to change to reduce the loss.

Mathematically, the gradient with respect to parameter w is the partial derivative:

```
dLoss/dw
```

If dLoss/dw > 0, increasing w increases the loss, so decrease w.
If dLoss/dw < 0, increasing w decreases the loss, so increase w.

**The update rule:**
```
w_new = w_old - lr * (dLoss/dw)
```

where **lr** is the **learning rate**: the size of the step.

## 2.4 Learning rate

![Learning rate](images/learning_rate.png)

*Too large: the model overshoots the valley and bounces around. Too small: progress
is so slow you will still be waiting next week. Just right: smooth convergence.*

The learning rate controls how much you adjust the parameters at each step.

**Too large (lr = 2.0):** giant steps on the mountain. You jump over the valley.
Loss oscillates or explodes.

**Too small (lr = 0.00001):** tiny steps. You will get there eventually, if
you have unlimited time and cloud budget.

**Just right (lr = 2e-4 in our project):** you reach the valley reasonably fast
without jumping past it.

There is no universal value. It is empirical. It is art. It is frustration.

We use lr = 1e-4 in pretraining and lr = 2e-4 in the GAN phase, with
**ReduceLROnPlateau**: if PSNR stops improving for 3 epochs, the learning rate
is automatically halved. The model "brakes" as it approaches convergence.

## 2.5 Backpropagation

![Backpropagation](images/backpropagation.png)

*Blue arrow: the forward pass. The input travels through the network and
produces a prediction. The loss compares prediction to ground truth.
Red arrows: the backward pass. Gradients flow right to left through every layer.
Each parameter receives its gradient and is updated.*

A neural network is a composed function: f(g(h(x))). To compute dLoss/dw for
a parameter buried deep in the network, we use the **chain rule** from calculus:

```
dLoss/dw = (dLoss/d_output) * (d_output/d_prev_layer) * ... * (d_layer/dw)
```

Backpropagation is the algorithm that applies this rule efficiently, computing
gradients from output back toward input (hence "back").

**Forward pass:** the image travels through the entire network, producing a prediction.
**Loss computation:** compare prediction to ground truth.
**Backward pass:** compute gradients for every parameter, from output toward input.
**Update:** adjust every parameter using its gradient.

This is one **training iteration**. You do this thousands of times.

## 2.6 Vanishing gradient

The chain rule multiplies many terms together. If each term is less than 1
(which happens with Sigmoid and Tanh for extreme values), their product shrinks
toward zero.

Result: parameters in early layers receive near-zero gradients and learn nothing.
The network grows deeper, but only the last few layers actually do anything useful.

Solutions:
- **ReLU:** gradient is 1 for positive inputs, does not shrink
- **Skip connections (ResNet):** provide a direct path for gradients, bypassing intermediate layers (Chapter 4)
- **Gradient clipping:** prevents the opposite problem, gradients exploding

## 2.7 Mini-batches

![Mini-batch gradient descent](images/minibatch.png)

*Left: a mini-batch gives a noisy estimate of the true loss curve, but that is
acceptable. Right: mini-batch converges faster per wall-clock time than full batch,
even though each step is noisier.*

Computing the gradient on all 10,400 training images before updating? That is
**full batch gradient descent**: slow and memory-intensive.

The alternative: pick a random subset of images (**mini-batch**), compute the
gradient on those, update, repeat.

We use **batch size = 16:** 16 images per iteration. Each 256x256 image is
roughly 200KB. 16 images fits comfortably on a T4 GPU.

One **epoch** is one complete pass through the dataset:
- 10,400 training images / batch size 16 = 650 iterations per epoch

## 2.8 Adam

Plain SGD (Stochastic Gradient Descent):
```
w = w - lr * gradient
```

Works, but is slow and sensitive to learning rate choices. Every parameter gets the
same step size, even if some dimensions need large steps and others need tiny ones.

**Adam** (Adaptive Moment Estimation, 2014) tracks two running statistics per
parameter:

- **m** (first moment): a moving average of recent gradients. This smooths out
  noise and gives a sense of the general direction the parameter is heading.
- **v** (second moment): a moving average of the squared gradients. This measures
  how much the gradient fluctuates. A noisy dimension has large v; a stable one
  has small v.

```
m = b1*m + (1-b1)*gradient          (b1 = 0.9 typically)
v = b2*v + (1-b2)*gradient^2        (b2 = 0.999 typically)
```

Think of b1=0.9 as saying: "90% of my memory, 10% new information." After many
steps, m is a weighted average of past gradients, with recent ones counting more.

**Why is the update rule a fraction (m_hat / sqrt(v_hat))?**

The division is the key idea. Consider two parameters:

- Parameter A: gradient is consistently around 5.0. So m is ~5.0 and v is ~25.
  Effective step: `lr * 5 / sqrt(25) = lr * 1.0`. Normal step.
- Parameter B: gradient jumps between +10 and -10 every step. So m averages near
  0 and v is large (~100). Effective step: `lr * 0 / sqrt(100)`, nearly zero.
  Adam does not blindly follow noisy signals.
- Parameter C: gradient is consistently tiny, 0.01. v is very small (~0.0001).
  Effective step: `lr * 0.01 / sqrt(0.0001) = lr * 1.0`. A full step, despite
  the tiny raw gradient. Adam does not ignore slow-moving dimensions.

The division by sqrt(v) normalizes each parameter's step to roughly the same
effective scale, regardless of how large or noisy its gradients are.

**What are m_hat and v_hat? (bias correction)**

Both m and v start at zero. In the first few steps they are too small: m has only
seen one or two gradients and is still pulled toward its initial value of 0. This
would make early steps too small.

The bias correction fixes this by dividing by how much the running average has
been "diluted" by the zero initialization:

```
m_hat = m / (1 - b1^t)     (t is the current step number)
v_hat = v / (1 - b2^t)
```

At step t=1 with b1=0.9: divides by (1 - 0.9^1) = 0.1, so m_hat = 10 * m.
This inflates the estimate back to what it would have been with no zero bias.

By step t=100: b1^100 is tiny, so (1 - b1^100) is close to 1, and m_hat = m.
The correction fades away as more real gradient history accumulates.

The full update:
```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

where `epsilon` (typically 1e-8) prevents division by zero if v_hat is 0.

Enough theory. Let's watch them both take a single step and see who embarrasses
themselves first.

**Setup:** same lr = 0.048, surface f = w1^2 + 20*w2^2, start at (w1=-1, w2=0.6).
Two parameters. One is on a gentle slope, one is on a cliff.

Gradients at the start: dL/dw1 = 2*(-1) = -2, dL/dw2 = 40*(0.6) = +24

SGD, first step:
```
w1 = -1.0 - 0.048 * (-2)  =  -1.0 + 0.096  =  -0.904   (good, small step)
w2 =  0.6 - 0.048 * (24)  =   0.6 - 1.152  =  -0.552   (overshot: was +0.6, now -0.552)
```
w2 flipped sign. Next step it flips again. It zigzags.

Adam, first step. The update rule is `w = w - lr * g / sqrt(v_hat)`.

`v_hat` is Adam's estimate of how large the gradient typically is on that parameter.
At t=1, v starts at 0, so the math simplifies cleanly:

```
v     = b2 * 0 + (1 - b2) * g^2  =  0.001 * g^2
v_hat = v / (1 - b2^1)           =  0.001 * g^2 / 0.001  =  g^2
```

The `/ (1 - b2^t)` bias correction happens to cancel perfectly at t=1.
So `v_hat = g^2`, which means `sqrt(v_hat) = |g|`.

The update becomes `w = w - lr * g / |g|`. The gradient divided by its own size
always gives exactly +1 or -1. So Adam takes a step of exactly `lr` in the
gradient direction, no matter how big or small the gradient was.

For w2 (gradient = +24, very large because the slope is steep):
```
sqrt(v_hat) = sqrt(24^2) = 24
actual step = lr * gradient / sqrt(v_hat) = 0.048 * 24 / 24 = 0.048
w2 = 0.6 - 0.048 = 0.552     (small controlled step, no overshoot)
```

For w1 (gradient = -2, small because the slope is gentle):
```
sqrt(v_hat) = sqrt((-2)^2) = 2
actual step = lr * |gradient| / sqrt(v_hat) = 0.048 * 2 / 2 = 0.048
w1 = -1.0 + 0.048 = -0.952   (same size step, even though gradient was 12x smaller)
```

Both parameters moved by exactly 0.048. SGD treated a cliff the same as a gentle
hill and fell off it. Adam saw the cliff, divided by the cliff, and took a normal
step. The gradient on w2 was 12 times larger than on w1, but Adam normalized each
one by itself, so neither dimension gets to throw a tantrum.

That is the whole mechanism: Adam cancels out gradient magnitude differences between
parameters, leaving only direction and the shared lr.

![Adam vs SGD](images/adam_vs_sgd.png)

*Same lr = 0.048, same start, same surface: f = w1^2 + 20*w2^2. The w2 axis is
20x steeper. SGD: the update factor on w2 is (1 - lr*40) = -0.92, so w2 flips
sign every step and zigzags. Adam: normalizes each gradient by the square root of
its running average of squared gradients, so the effective step on both axes is
approximately lr regardless of gradient magnitude. Step markers every 5 iterations.*

**Why b1 = 0.5 in our GAN?**

Standard Adam uses b1 = 0.9, which gives a long memory of past gradient directions.
In GAN training, the loss landscape shifts rapidly as generator and discriminator
evolve together. A long memory means the optimizer follows an old direction when
the landscape has already changed. Using b1 = 0.5 shortens the memory, allowing
faster responses to shifts in the adversarial dynamics.

## 2.9 Gradient clipping

![Gradient clipping](images/gradient_clipping.png)

*Left: a gradient spike at step 12 destroys training without clipping (red).
With clipping the spike is absorbed and training continues (green).
Right: each arrow is one gradient vector. Red dashed: original, too large,
outside the circle. Green solid: the same gradient after clipping, scaled back
to the boundary. The direction is identical. Only the length changed.*

Sometimes gradients explode: enormous values that send parameters completely off
track. This is especially common when unfreezing a frozen backbone mid-training,
exactly what happens in Phase 2 of our uncertainty model.

The right diagram shows what clipping does to a single gradient vector. Imagine
the model has millions of parameters, and each one has a gradient. Stack all those
gradients into one big vector. Compute its length (the "norm"). If that length
exceeds the threshold, shrink the entire vector proportionally until its length
equals the threshold. Every gradient gets scaled by the same factor, so their
relative sizes stay the same. The direction of the update does not change.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the gradient norm is 5.0 and max_norm is 1.0, all gradients are divided by 5.
Direction is preserved, magnitude is controlled.

We apply this after every backward pass on the generator during uncertainty training.

## 2.10 Putting it all together

```
1. Forward pass:   ab_pred = G(L)
2. Compute loss:   loss = L1(ab_pred, ab_real) * 100
3. Zero gradients: optimizer.zero_grad()
4. Backward pass:  loss.backward()
5. Clip gradients: clip_grad_norm_(G.parameters(), max_norm=1.0)
6. Update:         optimizer.step()
```

**Step 3: why zero the gradients?**

PyTorch does not clear gradients automatically between iterations. Every call to
`loss.backward()` adds the new gradients on top of whatever is already stored in
each parameter's `.grad` attribute. If you forget to zero them, the gradients from
iteration 1 are still there during iteration 2, and you end up applying the sum
of both iterations' gradients in the update. The model sees a mix of current and
stale information and behaves erratically.

`optimizer.zero_grad()` sets every `.grad` to zero before the new backward pass,
ensuring each update is based only on the current batch.

Repeat 650 times per epoch, 20 epochs per stage. That is 13,000 iterations
per training stage, and roughly 3.5 hours on a T4 GPU.

If the learning rate is wrong or the loss explodes? You start over.

## Summary

| Concept | What it means |
|---------|---------------|
| Loss function | A number measuring how wrong the model is right now |
| Gradient | Direction and magnitude of required change per parameter |
| Learning rate | The size of each update step |
| Backpropagation | Algorithm that computes gradients using the chain rule |
| Mini-batch | A subset of data per iteration (we use 16 images) |
| Epoch | One complete pass through the dataset |
| Adam | Adaptive optimizer that adjusts lr per parameter |
| Gradient clipping | Caps gradient magnitude to prevent training instability |

*Continue to [Chapter 3: Convolutional Networks](chapter_03_cnn.md)*
