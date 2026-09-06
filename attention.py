"""
Multi-head scaled dot-product attention, the position-wise feed-forward
block, and sinusoidal positional encoding -- the pieces that make a
Transformer a Transformer. Manual forward + backward, no autograd.
"""
import numpy as np
from layers import Module, Linear, softmax, softmax_backward


class PositionalEncoding:
    """Adds the classic sin/cos positional signal to token embeddings.
    Has no learnable parameters, so backward is just the identity."""

    def __init__(self, d_model, max_len=512):
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = pe

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[None, :seq_len, :]

    def backward(self, dout):
        return dout  # additive, no params -> gradient passes straight through


class MultiHeadAttention(Module):
    """
    Q, K, V each get their own linear projection, are split into `n_heads`
    parallel heads, attend independently, then get concatenated and mixed
    by one more linear layer.

    Shapes used throughout:
        x_q: (batch, len_q, d_model)   x_k, x_v: (batch, len_k, d_model)
        after splitting into heads:    (batch, n_heads, len, d_head)
    """

    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = Linear(d_model, d_model, rng=rng)
        self.w_k = Linear(d_model, d_model, rng=rng)
        self.w_v = Linear(d_model, d_model, rng=rng)
        self.w_o = Linear(d_model, d_model, rng=rng)

        self._cache = None

    def _split_heads(self, x):
        b, seq_len, _ = x.shape
        x = x.reshape(b, seq_len, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (b, heads, seq, d_head)

    def _merge_heads(self, x):
        b, h, seq_len, d_head = x.shape
        x = x.transpose(0, 2, 1, 3)  # (b, seq, heads, d_head)
        return x.reshape(b, seq_len, h * d_head)

    def forward(self, x_q, x_k, x_v, mask=None):
        """mask: broadcastable to (batch, 1, len_q, len_k), 0 where allowed
        and -1e9 where forbidden (padding / future tokens)."""
        Q = self._split_heads(self.w_q.forward(x_q))
        K = self._split_heads(self.w_k.forward(x_k))
        V = self._split_heads(self.w_v.forward(x_v))

        scale = 1.0 / np.sqrt(self.d_head)
        scores = np.einsum("bhqd,bhkd->bhqk", Q, K) * scale
        attn = softmax(scores, axis=-1, mask=mask)
        context = np.einsum("bhqk,bhkd->bhqd", attn, V)  # (b, h, len_q, d_head)

        merged = self._merge_heads(context)
        out = self.w_o.forward(merged)

        self._cache = (Q, K, V, attn, scale)
        return out

    def backward(self, dout):
        Q, K, V, attn, scale = self._cache

        d_merged = self.w_o.backward(dout)
        d_context = self._split_heads(d_merged)  # (b, h, len_q, d_head)

        # context = attn @ V
        d_attn = np.einsum("bhqd,bhkd->bhqk", d_context, V)
        dV = np.einsum("bhqk,bhqd->bhkd", attn, d_context)

        # attn = softmax(scores)
        d_scores = softmax_backward(d_attn, attn, axis=-1)

        # scores = Q @ K^T * scale
        dQ = np.einsum("bhqk,bhkd->bhqd", d_scores, K) * scale
        dK = np.einsum("bhqk,bhqd->bhkd", d_scores, Q) * scale

        dx_q = self.w_q.backward(self._merge_heads(dQ))
        dx_k = self.w_k.backward(self._merge_heads(dK))
        dx_v = self.w_v.backward(self._merge_heads(dV))
        return dx_q, dx_k, dx_v


class PositionwiseFeedForward(Module):
    """Two linear layers with a ReLU in between, applied independently at
    every sequence position: FFN(x) = ReLU(x W1 + b1) W2 + b2."""

    def __init__(self, d_model, d_ff, rng=None):
        self.fc1 = Linear(d_model, d_ff, rng=rng)
        self.fc2 = Linear(d_ff, d_model, rng=rng)
        self._relu_mask = None

    def forward(self, x):
        h = self.fc1.forward(x)
        relu_out = np.maximum(h, 0)
        self._relu_mask = (h > 0)
        return self.fc2.forward(relu_out)

    def backward(self, dout):
        d_relu = self.fc2.backward(dout)
        dh = d_relu * self._relu_mask
        return self.fc1.backward(dh)
