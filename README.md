# Transformer From Scratch (pure NumPy, manual backprop)

A full encoder-decoder Transformer ("Attention Is All You Need"), implemented
with **only NumPy** — no PyTorch/TensorFlow, no autograd. Every layer has a
hand-derived forward *and* backward pass, verified against numerical
gradients.

## Files

| File | Contents |
|---|---|
| `layers.py` | `Linear`, `LayerNorm`, `Embedding`, `softmax`, `Adam` optimizer |
| `attention.py` | `MultiHeadAttention`, `PositionwiseFeedForward`, `PositionalEncoding` |
| `model.py` | `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Transformer`, masks, loss |
| `test_gradients.py` | Numerical gradient checking for every component |
| `train_demo.py` | Trains the model on a toy sequence-reversal task end-to-end |

## Architecture

Standard post-LN Transformer, same shape as the original paper:

- Token embeddings scaled by `sqrt(d_model)` + sinusoidal positional encoding
- N encoder layers: multi-head self-attention → Add & Norm → feed-forward → Add & Norm
- N decoder layers: masked multi-head self-attention → Add & Norm → cross-attention
  over encoder output → Add & Norm → feed-forward → Add & Norm
- Final linear projection to target vocabulary + softmax cross-entropy loss
- Causal mask for decoder self-attention, padding masks for both encoder
  self-attention and cross-attention

Every module subclasses `Module`, which auto-collects `(name, param, grad)`
triples from itself and nested sub-modules/lists — that's what
`model.parameters()` feeds to the optimizer.

## Run it

```bash
# 1. Verify every backward pass is mathematically correct
python3 test_gradients.py

# 2. Train on the toy task and watch it converge
python3 train_demo.py
```

`test_gradients.py` compares analytic gradients (from the hand-written
`.backward()` methods) against finite-difference approximations for every
layer, and for several parameters deep inside a full forward/backward pass
of the whole model. All checks pass at ~1e-8–1e-10 relative error.

`train_demo.py` trains a small Transformer (d_model=64, 4 heads, 2 layers,
~170K params) to reverse sequences of 6 digits — a task that specifically
requires the decoder's cross-attention to look back at the *whole* encoded
input, not just copy position-to-position. In 400 Adam steps it goes from
~16% to ~97%+ token accuracy and decodes held-out examples correctly.

## Extending it

- Swap `train_demo.py`'s toy task for real tokenized text + a bigger
  `d_model`/`n_layers` to scale up.
- Add label smoothing, learning-rate warmup, or beam search decoding.
- Add dropout (forward: zero out + scale; backward: multiply by the same mask).
- For a decoder-only (GPT-style) model, drop the `Encoder` and
  `cross_attn` from `DecoderLayer` — the rest is unchanged.
