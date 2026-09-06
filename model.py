"""
Assembling the pieces into the full encoder-decoder Transformer
("Attention Is All You Need" architecture), plus masks and the loss.
"""
import numpy as np
from layers import Module, Linear, LayerNorm, Embedding, softmax
from attention import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


# ----------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------

def make_padding_mask(seq, pad_id):
    """seq: (batch, len). Returns (batch, 1, 1, len) with 0 for real tokens
    and -1e9 for padding, ready to broadcast-add onto attention scores."""
    mask = (seq == pad_id).astype(np.float64) * -1e9
    return mask[:, None, None, :]


def make_causal_mask(seq_len):
    """(1, 1, len, len) upper-triangular -1e9 mask so position i can only
    attend to positions <= i."""
    m = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return m[None, None, :, :]


def combine_masks(*masks):
    out = None
    for m in masks:
        out = m if out is None else out + m
    return out


# ----------------------------------------------------------------------
# Encoder / Decoder layers
# ----------------------------------------------------------------------

class EncoderLayer(Module):
    def __init__(self, d_model, n_heads, d_ff, rng=None):
        self.self_attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self._cache = None

    def forward(self, x, src_mask):
        attn_out = self.self_attn.forward(x, x, x, mask=src_mask)
        x1 = self.norm1.forward(x + attn_out)          # residual + post-LN
        ffn_out = self.ffn.forward(x1)
        x2 = self.norm2.forward(x1 + ffn_out)
        self._cache = (x, attn_out, x1, ffn_out)
        return x2

    def backward(self, dout):
        x, attn_out, x1, ffn_out = self._cache

        d_sum2 = self.norm2.backward(dout)
        d_x1_res = d_sum2                      # residual branch
        d_ffn_out = d_sum2                     # ffn branch
        d_x1_ffn = self.ffn.backward(d_ffn_out)
        d_x1 = d_x1_res + d_x1_ffn

        d_sum1 = self.norm1.backward(d_x1)
        d_x_res = d_sum1                       # residual branch
        d_attn_out = d_sum1                    # attn branch
        dq, dk, dv = self.self_attn.backward(d_attn_out)
        d_x_attn = dq + dk + dv                # x was used as q, k, and v
        dx = d_x_res + d_x_attn
        return dx


class DecoderLayer(Module):
    def __init__(self, d_model, n_heads, d_ff, rng=None):
        self.self_attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, rng=rng)
        self.norm3 = LayerNorm(d_model)
        self._cache = None

    def forward(self, x, enc_out, tgt_mask, cross_mask):
        self_attn_out = self.self_attn.forward(x, x, x, mask=tgt_mask)
        x1 = self.norm1.forward(x + self_attn_out)

        cross_attn_out = self.cross_attn.forward(x1, enc_out, enc_out, mask=cross_mask)
        x2 = self.norm2.forward(x1 + cross_attn_out)

        ffn_out = self.ffn.forward(x2)
        x3 = self.norm3.forward(x2 + ffn_out)

        self._cache = (x, x1, x2, enc_out)
        return x3

    def backward(self, dout):
        x, x1, x2, enc_out = self._cache

        d_sum3 = self.norm3.backward(dout)
        d_x2_res = d_sum3
        d_ffn_out = d_sum3
        d_x2_ffn = self.ffn.backward(d_ffn_out)
        d_x2 = d_x2_res + d_x2_ffn

        d_sum2 = self.norm2.backward(d_x2)
        d_x1_res = d_sum2
        d_cross_out = d_sum2
        dq, dk, dv = self.cross_attn.backward(d_cross_out)
        # dq flows back into the decoder stream (x1); dk, dv flow into the
        # encoder output (accumulated across all decoder layers).
        d_x1 = d_x1_res + dq
        d_enc_out = dk + dv

        d_sum1 = self.norm1.backward(d_x1)
        d_x_res = d_sum1
        d_self_out = d_sum1
        dq2, dk2, dv2 = self.self_attn.backward(d_self_out)
        d_x_self = dq2 + dk2 + dv2
        dx = d_x_res + d_x_self

        return dx, d_enc_out


# ----------------------------------------------------------------------
# Encoder / Decoder stacks
# ----------------------------------------------------------------------

class Encoder(Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=512, rng=None):
        self.embed = Embedding(vocab_size, d_model, rng=rng)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = [EncoderLayer(d_model, n_heads, d_ff, rng=rng) for _ in range(n_layers)]
        self.d_model = d_model
        self._embed_scale = np.sqrt(d_model)

    def forward(self, src_ids, src_mask):
        x = self.embed.forward(src_ids) * self._embed_scale
        x = self.pos_enc.forward(x)
        for layer in self.layers:
            x = layer.forward(x, src_mask)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        dout = self.pos_enc.backward(dout)
        self.embed.backward(dout * self._embed_scale)


class Decoder(Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=512, rng=None):
        self.embed = Embedding(vocab_size, d_model, rng=rng)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = [DecoderLayer(d_model, n_heads, d_ff, rng=rng) for _ in range(n_layers)]
        self.d_model = d_model
        self._embed_scale = np.sqrt(d_model)

    def forward(self, tgt_ids, enc_out, tgt_mask, cross_mask):
        x = self.embed.forward(tgt_ids) * self._embed_scale
        x = self.pos_enc.forward(x)
        for layer in self.layers:
            x = layer.forward(x, enc_out, tgt_mask, cross_mask)
        return x

    def backward(self, dout):
        d_enc_total = 0.0
        for layer in reversed(self.layers):
            dout, d_enc_out = layer.backward(dout)
            d_enc_total = d_enc_total + d_enc_out
        dout = self.pos_enc.backward(dout)
        self.embed.backward(dout * self._embed_scale)
        return d_enc_total


# ----------------------------------------------------------------------
# Full Transformer
# ----------------------------------------------------------------------

class Transformer(Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, n_heads=4,
                 d_ff=128, n_layers=2, max_len=512, seed=0):
        rng = np.random.default_rng(seed)
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, max_len, rng)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, n_layers, max_len, rng)
        self.out_proj = Linear(d_model, tgt_vocab_size, rng=rng)
        self._cache = None

    def forward(self, src_ids, tgt_in_ids, pad_id=0):
        src_mask = make_padding_mask(src_ids, pad_id)
        causal = make_causal_mask(tgt_in_ids.shape[1])
        tgt_pad = make_padding_mask(tgt_in_ids, pad_id)
        tgt_mask = combine_masks(causal, tgt_pad)
        cross_mask = src_mask  # decoder attends to encoder outputs; mask out src padding

        enc_out = self.encoder.forward(src_ids, src_mask)
        dec_out = self.decoder.forward(tgt_in_ids, enc_out, tgt_mask, cross_mask)
        logits = self.out_proj.forward(dec_out)
        return logits

    def backward(self, dlogits):
        d_dec_out = self.out_proj.backward(dlogits)
        d_enc_out = self.decoder.backward(d_dec_out)
        self.encoder.backward(d_enc_out)

    @staticmethod
    def loss_and_grad(logits, targets, pad_id=0):
        """Softmax cross-entropy averaged over non-pad tokens.
        logits: (batch, len, vocab), targets: (batch, len) int ids."""
        b, t, v = logits.shape
        probs = softmax(logits, axis=-1)
        mask = (targets != pad_id).astype(np.float64)
        n_tokens = max(mask.sum(), 1.0)

        flat_probs = probs.reshape(-1, v)
        flat_targets = targets.reshape(-1)
        idx = np.arange(flat_targets.shape[0])
        correct_probs = np.clip(flat_probs[idx, flat_targets], 1e-12, 1.0)
        loss = -np.log(correct_probs).reshape(b, t)
        loss = (loss * mask).sum() / n_tokens

        dlogits = probs.copy()
        dlogits.reshape(-1, v)[idx, flat_targets] -= 1.0
        dlogits = dlogits * mask[:, :, None] / n_tokens
        return loss, dlogits

    def greedy_decode(self, src_ids, sos_id, eos_id, pad_id=0, max_len=20):
        """Autoregressive greedy decoding (inference only, no gradients)."""
        src_mask = make_padding_mask(src_ids, pad_id)
        enc_out = self.encoder.forward(src_ids, src_mask)
        batch = src_ids.shape[0]
        tgt = np.full((batch, 1), sos_id, dtype=int)

        for _ in range(max_len - 1):
            causal = make_causal_mask(tgt.shape[1])
            dec_out = self.decoder.forward(tgt, enc_out, causal, src_mask)
            logits = self.out_proj.forward(dec_out)
            next_ids = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            tgt = np.concatenate([tgt, next_ids], axis=1)
            if np.all(next_ids == eos_id):
                break
        return tgt
