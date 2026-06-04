# Neural Networks: Zero to Hero — Study & Implementation

A from-scratch, deeply-annotated walkthrough of Andrej Karpathy's
[**Neural Networks: Zero to Hero**](https://github.com/karpathy/nn-zero-to-hero) series.

This is my learning log. Instead of copying the reference code, I **re-implement each
component by hand**, annotate *why* every line works, verify my gradients against PyTorch,
and solve the exercises. The goal is a first-principles understanding of how neural networks
— from automatic differentiation up to language models — actually work under the hood.

## What's inside

### `micrograd/` — a tiny autograd engine
- **`micrograd.ipynb`** — A `Value` class that builds a computational graph during the forward
  pass, with **reverse-mode backprop** via topological sort and correct gradient *accumulation*
  (a bug I hit and fixed — see commit history). Built up into a `Neuron → Layer → MLP` and a full
  **training loop** (forward → zero-grad → backward → SGD), with gradients **cross-checked against PyTorch**.
- **`micrograd_exercises.ipynb`** — Analytical, numerical, and symmetric-derivative gradients;
  extending `Value` with **softmax + negative log-likelihood** (`exp`/`log`/`div` from scratch),
  every gradient **verified against `torch`**.

### `makemore/` — a character-level language model
- **`makemore.ipynb`** — The *same* bigram model built two ways:
  - **Statistical**: a 27×27 count matrix normalized into distributions, sampled with
    `torch.multinomial`, with Laplace smoothing.
  - **Neural net**: the same problem as one linear layer — one-hot inputs, logits as **log-counts**,
    **softmax** derived from first principles, **negative log-likelihood** loss derived step by step,
    and **gradient descent** — showing L2 regularization *is* the smoothing from the statistical model.
  - Annotated throughout on **broadcasting**, tensor shapes, and what each `@` computes.

## Key concepts

Reverse-mode autodiff · computational graphs · analytical vs. numerical gradients ·
softmax & cross-entropy derived by hand · verifying custom gradients against PyTorch ·
maximum-likelihood estimation · a statistical model recast as an equivalent neural net ·
broadcasting & matrix shapes · regularization as smoothing.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # numpy, matplotlib, graphviz, torch
jupyter notebook                   # open any .ipynb and run top-to-bottom
```
> Graph visualizations need the Graphviz binary (`brew install graphviz`).

## Progress

- [x] **micrograd** — autograd engine, MLP, training loop
- [x] **micrograd exercises** — gradients & softmax, verified against PyTorch
- [x] **makemore (bigram)** — statistical + single-layer neural model
- [ ] **makemore** — MLP, BatchNorm, WaveNet
- [ ] **Building GPT** — self-attention and the Transformer from scratch
- [ ] **Tokenization (minBPE)** & **GPT-2**

---

All exercises and the lecture series are by
**[Andrej Karpathy](https://github.com/karpathy/nn-zero-to-hero)**; the implementations,
notes, and solutions here are my own.
