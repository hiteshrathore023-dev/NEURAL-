"""
Numerical gradient checking. For every layer, and for the full model, we
compare the analytically-computed gradient (our hand-written backward pass)
against a finite-difference approximation. This is the standard way to
prove a from-scratch backward implementation is actually correct.
"""
import numpy as np
from layers import Linear, LayerNorm, Embedding
from attention import MultiHeadAttention, PositionwiseFeedForward
from model import Transformer, make_causal_mask

np.random.seed(0)


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f()
        x[idx] = orig - eps
        minus = f()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check(name, analytic, numeric, tol=1e-4):
    err = rel_error(analytic, numeric)
    status = "OK" if err < tol else "FAIL"
    print(f"[{status}] {name:35s} max relative error = {err:.2e}")
    return err < tol


def scalar_loss(out):
    # A fixed, arbitrary "loss" so every output element gets a nonzero, varied gradient.
    rng = np.random.default_rng(42)
    weights = rng.normal(size=out.shape)
    return np.sum(out * weights), weights


all_ok = True

# ---------------------------------------------------------------- Linear
x = np.random.randn(3, 5, 8)
lin = Linear(8, 6)
out = lin.forward(x)
loss, w = scalar_loss(out)
lin.backward(w)
def f():
    return np.sum(lin.forward(x) * w)
all_ok &= check("Linear dW", lin.grads["W"], numerical_grad(f, lin.params["W"]))
all_ok &= check("Linear db", lin.grads["b"], numerical_grad(f, lin.params["b"]))
grad_x = numerical_grad(f, x)
lin.grads = {k: np.zeros_like(v) for k, v in lin.params.items()}
out = lin.forward(x)
dx = lin.backward(w)
all_ok &= check("Linear dx", dx, grad_x)

# ---------------------------------------------------------------- LayerNorm
x = np.random.randn(2, 4, 8)
ln = LayerNorm(8)
out = ln.forward(x)
loss, w = scalar_loss(out)
def f():
    return np.sum(ln.forward(x) * w)
dx = ln.backward(w)
all_ok &= check("LayerNorm dx", dx, numerical_grad(f, x))
all_ok &= check("LayerNorm dgamma", ln.grads["gamma"], numerical_grad(f, ln.params["gamma"]))
all_ok &= check("LayerNorm dbeta", ln.grads["beta"], numerical_grad(f, ln.params["beta"]))

# ---------------------------------------------------------------- Embedding
ids = np.array([[1, 2, 3], [3, 2, 0]])
emb = Embedding(5, 6)
out = emb.forward(ids)
loss, w = scalar_loss(out)
def f():
    return np.sum(emb.forward(ids) * w)
emb.backward(w)
all_ok &= check("Embedding dtable", emb.grads["table"], numerical_grad(f, emb.params["table"]))

# ---------------------------------------------------------------- MultiHeadAttention
x = np.random.randn(2, 4, 8)
mha = MultiHeadAttention(8, 2)
mask = make_causal_mask(4)
out = mha.forward(x, x, x, mask=mask)
loss, w = scalar_loss(out)
def f():
    return np.sum(mha.forward(x, x, x, mask=mask) * w)
dq, dk, dv = mha.backward(w)
dx_total = dq + dk + dv  # x feeds all three inputs here
all_ok &= check("MultiHeadAttention dx (q=k=v)", dx_total, numerical_grad(f, x))
all_ok &= check("MultiHeadAttention w_q.dW", mha.w_q.grads["W"], numerical_grad(f, mha.w_q.params["W"]))
all_ok &= check("MultiHeadAttention w_o.dW", mha.w_o.grads["W"], numerical_grad(f, mha.w_o.params["W"]))

# ---------------------------------------------------------------- PositionwiseFeedForward
x = np.random.randn(2, 4, 8)
ffn = PositionwiseFeedForward(8, 16)
out = ffn.forward(x)
loss, w = scalar_loss(out)
def f():
    return np.sum(ffn.forward(x) * w)
dx = ffn.backward(w)
all_ok &= check("FeedForward dx", dx, numerical_grad(f, x))
all_ok &= check("FeedForward fc1.dW", ffn.fc1.grads["W"], numerical_grad(f, ffn.fc1.params["W"]))

# ---------------------------------------------------------------- Full Transformer (loss w.r.t. embeddings)
model = Transformer(src_vocab_size=10, tgt_vocab_size=10, d_model=8, n_heads=2,
                     d_ff=16, n_layers=2, seed=1)
src = np.array([[1, 2, 3, 4]])
tgt_in = np.array([[1, 2, 3]])
tgt_out = np.array([[2, 3, 4]])

logits = model.forward(src, tgt_in, pad_id=0)
loss, dlogits = model.loss_and_grad(logits, tgt_out, pad_id=0)
model.backward(dlogits)

# Check one representative parameter deep in the stack: encoder layer 0's Q projection.
target_param = model.encoder.layers[0].self_attn.w_q.params["W"]
target_grad = model.encoder.layers[0].self_attn.w_q.grads["W"]

def full_loss():
    logits = model.forward(src, tgt_in, pad_id=0)
    l, _ = model.loss_and_grad(logits, tgt_out, pad_id=0)
    return l

all_ok &= check("Full model: encoder L0 w_q.dW", target_grad, numerical_grad(full_loss, target_param))

# Also check the output projection and target embedding table.
target_param2 = model.out_proj.params["W"]
target_grad2 = model.out_proj.grads["W"]
all_ok &= check("Full model: out_proj.dW", target_grad2, numerical_grad(full_loss, target_param2))

target_param3 = model.decoder.embed.params["table"]
target_grad3 = model.decoder.embed.grads["table"]
all_ok &= check("Full model: decoder embed.dtable", target_grad3, numerical_grad(full_loss, target_param3))

print()
print("ALL GRADIENT CHECKS PASSED" if all_ok else "SOME GRADIENT CHECKS FAILED")
