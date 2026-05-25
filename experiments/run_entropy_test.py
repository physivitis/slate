"""Reproduce the information monotonicity test (Patent Claim 17 Part B).

Builds the C3-strict architecture, applies spectral projection, then
measures input and output entropy of every compression layer using two
independent estimators. Verifies H(G(X)) <= H(X) by data processing inequality.

Expected result: 0 / 16 layers violate monotonicity under either estimator.

Run:
  python experiments/run_entropy_test.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slate import SlateGPT, gaussian_entropy, spectral_entropy  # noqa: E402

# Reuse data loading from C3 script
sys.path.insert(0, os.path.dirname(__file__))
from run_c3_strict import load_data, make_batch_fn, SEQ_LEN  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def main():
    torch.manual_seed(SEED)

    _train_data, val_data, vocab_size = load_data()
    # Use a larger batch for stable entropy estimates
    def get_batch():
        ix = torch.randint(len(val_data) - SEQ_LEN - 1, (64,))
        x = torch.stack([val_data[i:i+SEQ_LEN] for i in ix])
        return x.to(DEVICE)

    model = SlateGPT(vocab_size=vocab_size, seq_len=SEQ_LEN).to(DEVICE)
    # Apply spectral projection as in C3-strict
    model.project_constraints(target_max=1.0)
    model.eval()

    sigmas = model.measure_constraints()
    max_sigma = max(s for _, s in sigmas)
    print(f"=== Spectral constraint check ===")
    print(f"Max sigma_1 across {len(sigmas)} constrained layers: {max_sigma:.6f}")
    assert max_sigma <= 1.0001, "Spectral constraint not enforced!"
    print("Constraint enforced.\n")

    # Hook inputs and outputs of all constrained layers
    captures = {}

    def make_hook(name):
        def hook(_module, inputs, output):
            captures[name] = {
                "input": inputs[0].detach().float().cpu(),
                "output": output.detach().float().cpu(),
            }
        return hook

    hooks = [m.register_forward_hook(make_hook(name)) for name, m in model.constrained_layers]
    try:
        with torch.no_grad():
            _ = model(get_batch())
    finally:
        for h in hooks:
            h.remove()

    print("=== Information monotonicity test (Patent Claim 17 Part B) ===\n")
    print(f"{'Layer':<35} {'H_in (G)':<12} {'H_out (G)':<12} {'Delta':<10} "
          f"{'H_in (S)':<12} {'H_out (S)':<12} {'Status'}")
    print("=" * 110)

    v_g = v_s = 0
    for name, cap in captures.items():
        _tot_in_g, h_in_g = gaussian_entropy(cap["input"])
        _tot_out_g, h_out_g = gaussian_entropy(cap["output"])
        delta = h_out_g - h_in_g
        h_in_s, _, _ = spectral_entropy(cap["input"])
        h_out_s, _, _ = spectral_entropy(cap["output"])
        mono_gauss = delta <= 0.01
        mono_spec = h_out_s <= h_in_s + 0.01
        if not mono_gauss:
            v_g += 1
        if not mono_spec:
            v_s += 1
        status = "MONOTONIC" if (mono_gauss and mono_spec) else "VIOLATION"
        print(f"{name:<35} {h_in_g:<12.4f} {h_out_g:<12.4f} {delta:<+10.4f} "
              f"{h_in_s:<12.4f} {h_out_s:<12.4f} {status}")

    print(f"\nGaussian violations: {v_g} / {len(captures)}")
    print(f"Spectral violations: {v_s} / {len(captures)}")

    print(f"\n=== PATENT CLAIM 17 STATUS ===")
    print(f"Part A (sigma_1 <= 1):     {'VALIDATED' if max_sigma <= 1.0001 else 'FAIL'}")
    print(f"Part B (Gaussian DPI):     {'VALIDATED' if v_g == 0 else f'FAIL ({v_g})'}")
    print(f"Part B (spectral DPI):     {'VALIDATED' if v_s == 0 else f'FAIL ({v_s})'}")


if __name__ == "__main__":
    main()
