# How to Colorize the World with Mathematics
### A complete guide to AI for normal people (and a few abnormal ones)

> *"Any sufficiently advanced technology is indistinguishable from magic."*
> — Arthur C. Clarke
>
> *"Sure, but this magic runs on a GPU and costs pennies per hour on Colab."*
> — Every AI student, ever

This book grew out of one seemingly simple question:

**Can a computer colorize a black-and-white photograph?**

The answer is yes. But the road to get there passes through artificial neurons,
networks that fight each other, loss functions with dramatic names, and training
runs that take hours on a GPU you do not own.

Every concept here is explained from scratch, with examples, with a bit of humor,
and with exactly as much math as is necessary. No more.

All concepts connect directly to the real implementation in this repository: a
GAN-based image colorization system using a ResNet-18 U-Net generator, trained
on 13,000 COCO images.

**Scope note.** This book walks through the baseline pipeline implemented in
`src/`: ResNet-18 U-Net generator, PatchGAN discriminator, L1 + adversarial +
TV loss. That is the foundation, and every concept here (LAB color space,
skip connections, the GAN game, loss balancing, training dynamics) applies
directly to the final system too. The best-performing model reported in the
thesis builds on top of this baseline with three additions not covered here:
a colorfulness-based training curriculum, a semantic color consistency loss
(SCCL, built on frozen DINOv2 features), and a second DINOv2 cross-attention
refinement stage, trained on a larger 15,000-image split. That pipeline lives
in `notebooks/04_main_models/` rather than in `src/` - see the top-level
[README](../README.md) and `paper/thesis.pdf` for its architecture and results.

---

## Chapters

| Chapter | Title |
|---------|-------|
| 1 | [What on Earth Is an Artificial Neuron?](chapter_01_the_neuron.md) |
| 2 | [How a Model Learns](chapter_02_learning.md) |
| 3 | [Convolutional Networks — How a Computer Sees](chapter_03_cnn.md) |
| 4 | [ResNet — The Network with Shortcuts](chapter_04_resnet.md) |
| 5 | [U-Net — The Network with Memory](chapter_05_unet.md) |
| 6 | [Color Space — Why Not RGB?](chapter_06_colors.md) |
| 7 | [GANs — The Greatest Duel in AI](chapter_07_gan.md) |
| 8 | [Loss Functions — Teaching a Model What "Good" Means](chapter_08_loss.md) |
| 9 | [Uncertainty — When the Model Knows It Does Not Know](chapter_09_uncertainty.md) |
| 10 | [Training — The GPU Marathon](chapter_10_training.md) |

---

*Diagrams are in the [images/](images/) folder.*
