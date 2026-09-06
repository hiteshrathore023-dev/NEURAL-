"""
Demo: train the from-scratch Transformer on a toy sequence-reversal task.

    input:  5 8 2 9 3
    target: 3 9 2 8 5

This is a standard smoke test for seq2seq Transformers: it needs the
encoder to represent the whole source sequence, and the decoder to use
cross-attention (not just memorize position i -> position i).
"""
import numpy as np
from model import Transformer

rng = np.random.default_rng(0)

# --- Vocabulary: digits 1-9 as "content" tokens, plus specials. ---
PAD, SOS, EOS = 0, 10, 11
VOCAB_SIZE = 12
SEQ_LEN = 6


def make_batch(batch_size):
    src = rng.integers(1, 10, size=(batch_size, SEQ_LEN))
    reversed_seq = src[:, ::-1]
    tgt_in = np.concatenate([np.full((batch_size, 1), SOS), reversed_seq[:, :-1]], axis=1)
    tgt_out = reversed_seq
    return src, tgt_in, tgt_out


def accuracy(logits, tgt_out):
    preds = np.argmax(logits, axis=-1)
    return np.mean(preds == tgt_out)


def main():
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=32, seed=0,
    )
    optimizer_params = list(model.parameters())
    from layers import Adam
    opt = Adam(optimizer_params, lr=3e-3)

    print(f"Total parameters: {sum(p.size for _, p, _ in optimizer_params):,}\n")

    n_steps = 400
    batch_size = 32
    for step in range(1, n_steps + 1):
        src, tgt_in, tgt_out = make_batch(batch_size)

        logits = model.forward(src, tgt_in, pad_id=PAD)
        loss, dlogits = model.loss_and_grad(logits, tgt_out, pad_id=PAD)

        opt.zero_grad()
        model.backward(dlogits)
        opt.step()

        if step % 20 == 0 or step == 1:
            acc = accuracy(logits, tgt_out)
            print(f"step {step:4d}  loss {loss:.4f}  token-acc {acc:.3f}")

    print("\n--- Greedy decoding on fresh examples ---")
    src, _, tgt_out = make_batch(5)
    decoded = model.greedy_decode(src, sos_id=SOS, eos_id=EOS, pad_id=PAD, max_len=SEQ_LEN + 1)
    for i in range(5):
        pred = decoded[i, 1:1 + SEQ_LEN]  # drop the leading SOS
        print(f"input:  {src[i].tolist()}")
        print(f"target: {tgt_out[i].tolist()}")
        print(f"output: {pred.tolist()}")
        print(f"{'match' if np.array_equal(pred, tgt_out[i]) else 'DIFFERS'}\n")


if __name__ == "__main__":
    main()
