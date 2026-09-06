"""
Core building-block layers for a from-scratch Transformer.

Every layer is implemented with plain NumPy: it stores its own parameters,
computes a forward pass while caching what it needs, and computes a full
backward pass (gradients w.r.t. inputs AND parameters) by hand. There is no
autograd anywhere in this project -- every derivative below was worked out
with calculus and verified against numerical gradients (see test_gradients.py).
"""
import numpy as np


class Module:
    """Base class: knows how to collect its own (param, grad) pairs,
    including those of any sub-modules it owns."""

    def parameters(self):
        """Yield (name, param_array, grad_array) triples for this module
        and all sub-modules stored as attributes."""
        params = getattr(self, "params", {})
        grads = getattr(self, "grads", {})
        for name in params:
            yield (f"{self.__class__.__name__}.{name}", params[name], grads[name])
        for attr_name, attr in vars(self).items():
            if isinstance(attr, Module):
                for p in attr.parameters():
                    yield p
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        for p in item.parameters():
                            yield p

    def zero_grad(self):
        for _, _, g in self.parameters():
            g[...] = 0.0


class Linear(Module):
    """y = x @ W + b, applied to the last dimension of x."""

    def __init__(self, in_features, out_features, bias=True, rng=None):
        rng = rng or np.random.default_rng()
        # Xavier/Glorot init keeps activations well-scaled at the start.
        limit = np.sqrt(6.0 / (in_features + out_features))
        W = rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
        b = np.zeros(out_features, dtype=np.float64)
        self.params = {"W": W, "b": b} if bias else {"W": W}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.use_bias = bias
        self._x = None

    def forward(self, x):
        self._x = x
        out = x @ self.params["W"]
        if self.use_bias:
            out = out + self.params["b"]
        return out

    def backward(self, dout):
        x = self._x
        # Flatten all leading (batch/seq) dims so this works for any input rank.
        flat_x = x.reshape(-1, x.shape[-1])
        flat_dout = dout.reshape(-1, dout.shape[-1])
        self.grads["W"][...] += flat_x.T @ flat_dout
        if self.use_bias:
            self.grads["b"][...] += flat_dout.sum(axis=0)
        dx = dout @ self.params["W"].T
        return dx


class LayerNorm(Module):
    """Normalizes over the last dimension, then applies a learned scale/shift."""

    def __init__(self, dim, eps=1e-6):
        self.params = {"gamma": np.ones(dim), "beta": np.zeros(dim)}
        self.grads = {"gamma": np.zeros(dim), "beta": np.zeros(dim)}
        self.eps = eps
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mu) * std_inv
        out = self.params["gamma"] * x_hat + self.params["beta"]
        self._cache = (x_hat, std_inv)
        return out

    def backward(self, dout):
        x_hat, std_inv = self._cache
        gamma = self.params["gamma"]
        N = dout.shape[-1]

        self.grads["gamma"][...] += (dout * x_hat).reshape(-1, N).sum(axis=0)
        self.grads["beta"][...] += dout.reshape(-1, N).sum(axis=0)

        dx_hat = dout * gamma
        # Standard LayerNorm backward formula (derived from d(mean)/dx and d(var)/dx):
        dx = std_inv / N * (
            N * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return dx


class Embedding(Module):
    """Lookup table: integer ids -> d_model vectors."""

    def __init__(self, vocab_size, d_model, rng=None):
        rng = rng or np.random.default_rng()
        table = rng.normal(0, d_model ** -0.5, size=(vocab_size, d_model))
        self.params = {"table": table}
        self.grads = {"table": np.zeros_like(table)}
        self._ids = None

    def forward(self, ids):
        self._ids = ids
        return self.params["table"][ids]

    def backward(self, dout):
        # Scatter-add: multiple positions can reference the same row.
        np.add.at(self.grads["table"], self._ids, dout)
        return None  # ids are not differentiable


def softmax(x, axis=-1, mask=None):
    """Numerically stable softmax. `mask` (same shape, broadcastable) has
    -inf (or a very negative number) where attention must not attend."""
    if mask is not None:
        x = x + mask
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_backward(dy, y, axis=-1):
    """Backward of the softmax above. y is the softmax OUTPUT (cached)."""
    return y * (dy - np.sum(dy * y, axis=axis, keepdims=True))


class Adam:
    """Textbook Adam optimizer operating directly on the (param, grad) pairs
    collected from a model via `model.parameters()`."""

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.98), eps=1e-9):
        self.params = list(parameters)  # list of (name, param, grad)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for _, p, _ in self.params]
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self):
        self.t += 1
        for i, (_, p, g) in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for _, _, g in self.params:
            g[...] = 0.0
