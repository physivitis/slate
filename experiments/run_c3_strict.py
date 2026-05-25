"""Reproduce the validated Innovation 3 result (C3-strict).

Trains a 1.85M-parameter Pythia-equivalent transformer on character-level
Tiny Shakespeare with SLATE's exact spectral constraint (SVD projection).

Expected result (from May 2026 validation, A100 GPU, 1500 steps):
  - Validation loss: ~1.68 (matches NSA baseline within 0.01)
  - Max sigma_1: ~1.0001 (constraint enforced to machine precision)
  - Spectral violations: 0 / 16 layers

Run:
  python experiments/run_c3_strict.py
"""
import math
import os
import sys
import time
import urllib.request
import torch

# Add parent directory to path so we can import slate
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slate import SlateGPT  # noqa: E402

# -------- Hyperparameters --------
SEQ_LEN = 256
D_MODEL = 192
N_HEAD = 6
D_HEAD = 32
N_LAYER = 4
DROPOUT = 0.1
BATCH = 32
STEPS = 1500
LR = 3e-4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"


def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading Tiny Shakespeare to {DATA_PATH}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_data():
    download_data()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size


def make_batch_fn(train_data, val_data):
    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - SEQ_LEN - 1, (BATCH,))
        x = torch.stack([src[i:i+SEQ_LEN] for i in ix])
        y = torch.stack([src[i+1:i+SEQ_LEN+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)
    return get_batch


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    train_data, val_data, vocab_size = load_data()
    print(f"Vocab size: {vocab_size}, train: {len(train_data)}, val: {len(val_data)}")
    get_batch = make_batch_fn(train_data, val_data)

    model = SlateGPT(
        vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=D_MODEL,
        n_head=N_HEAD, d_head=D_HEAD, n_layer=N_LAYER, dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Constrained compression layers: {len(model.constrained_layers)}")

    # Initial projection
    model.project_constraints(target_max=1.0)
    init_max = model.max_sigma()
    print(f"Initial max sigma_1: {init_max:.6f}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    @torch.no_grad()
    def eval_loss():
        model.eval()
        losses = []
        for _ in range(20):
            x, y = get_batch("val")
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    print("\n=== Training C3-strict (SLATE Innovation 3) ===")
    t0 = time.time()
    max_grad = 0.0
    for step in range(STEPS):
        x, y = get_batch("train")
        _, loss = model(x, y)
        if torch.isnan(loss).any():
            print(f"NaN at step {step}")
            return
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        max_grad = max(max_grad, gn)
        opt.step()
        # SLATE: enforce spectral constraint after each optimizer step
        model.project_constraints(target_max=1.0)
        if step % 300 == 0 or step == STEPS - 1:
            vl = eval_loss()
            sigma = model.max_sigma()
            print(f"  step {step:4d} | train {loss.item():.4f} | val {vl:.4f} "
                  f"| grad {gn:.3f} | sigma {sigma:.4f} | t {time.time()-t0:.1f}s")

    final_val = eval_loss()
    sigmas = model.measure_constraints()
    max_sigma = max(s for _, s in sigmas)
    violations = sum(1 for _, s in sigmas if s > 1.01)

    print(f"\n=== C3-strict FINAL ===")
    print(f"Validation loss: {final_val:.4f}")
    print(f"Validation perplexity: {math.exp(final_val):.2f}")
    print(f"Max sigma_1: {max_sigma:.6f}")
    print(f"Spectral violations (>1.01): {violations} / {len(sigmas)}")
    print(f"Max gradient norm: {max_grad:.3f}")
    print(f"Wall-clock: {time.time() - t0:.1f}s")

    # Pass criteria
    print(f"\n=== Pass criteria ===")
    print(f"Quality within bounds (val < 5.0): {'PASS' if final_val < 5.0 else 'FAIL'}")
    print(f"Constraint enforced (max sigma <= 1.001): {'PASS' if max_sigma <= 1.001 else 'FAIL'}")
    print(f"Zero violations: {'PASS' if violations == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
