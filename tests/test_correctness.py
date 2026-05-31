"""
SLATE correctness test harness (CPU, runs anywhere).

Mechanism-correctness checks for the SLATE package. These confirm the
implementation does what it claims MATHEMATICALLY. They are unit tests,
not scientific validation: they verify the linear-algebra-level properties
that hold regardless of scale.

Run:  python tests/test_correctness.py
(or with pytest:  pytest tests/test_correctness.py -v)

What these prove:
  - SVD projection enforces sigma_1(G) <= 1 to machine precision
  - A spectrally-constrained operator does not increase entropy (DPI)
  - The stochastic eigenvalue estimator returns finite, sane values

What these do NOT prove:
  - Anything about production scale, quality, or advantage. That requires
    the GPU experiments in the production validation protocol.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from slate.spectral import project_to_unit_ball, measure_sigma1, find_compress_linears
from slate.utils import gaussian_entropy, spectral_entropy
from slate.lyapunov import stochastic_eigenvalue

torch.manual_seed(0)
PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


print("=" * 64)
print("SLATE CORRECTNESS HARNESS")
print("=" * 64)

# ---- Test 1: SVD projection enforces the spectral constraint ----
print("\nTest 1: SVD projection enforces sigma_1(G) <= 1")
layers = []
for i in range(8):
    lin = nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        lin.weight.mul_(3.0)  # deliberately inflate so sigma_1 >> 1
    layers.append((f"layer{i}", lin))

before = max(s for _, s in measure_sigma1(layers))
project_to_unit_ball(layers, target_max=1.0)
after = measure_sigma1(layers)
max_after = max(s for _, s in after)

check("sigma_1 inflated before projection", before > 2.0, f"max sigma_1 = {before:.3f}")
check("sigma_1 <= 1.0001 after projection", max_after <= 1.0001, f"max sigma_1 = {max_after:.6f}")
check("constraint holds on ALL layers", all(s <= 1.0001 for _, s in after),
      f"{sum(1 for _,s in after if s>1.0001)} violations / {len(after)}")

# ---- Test 2: Information monotonicity through constrained operator ----
print("\nTest 2: H(G(X)) <= H(X) for spectrally-constrained G (data processing inequality)")
g_viol = s_viol = 0
n_layers = 16
for i in range(n_layers):
    G = nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        G.weight.mul_(2.5)
    project_to_unit_ball([("G", G)], target_max=1.0)
    x = torch.randn(512, 64) * (1.0 + 0.5 * i / n_layers)  # varied input scale
    with torch.no_grad():
        gx = G(x)
    _, h_in_g = gaussian_entropy(x)
    _, h_out_g = gaussian_entropy(gx)
    h_in_s, _, _ = spectral_entropy(x)
    h_out_s, _, _ = spectral_entropy(gx)
    if h_out_g > h_in_g + 0.01:
        g_viol += 1
    if h_out_s > h_in_s + 0.01:
        s_viol += 1

check("Gaussian estimator: zero monotonicity violations", g_viol == 0,
      f"{g_viol} / {n_layers} layers")
check("Spectral estimator: zero monotonicity violations", s_viol == 0,
      f"{s_viol} / {n_layers} layers")

# ---- Test 3: Stochastic eigenvalue estimator returns sane values ----
print("\nTest 3: Lyapunov Hessian eigenvalue estimator returns finite, sane values")

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 16)
        self.l2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))

net = TinyNet()
xb = torch.randn(32, 16)
yb = torch.randint(0, 4, (32,))

def loss_fn(model, x, y):
    return nn.functional.cross_entropy(model(x), y)

vals = [stochastic_eigenvalue(net, (xb, yb), loss_fn) for _ in range(5)]
finite = all(math.isfinite(v) for v in vals)
bounded = all(abs(v) < 1e4 for v in vals)
check("eigenvalue estimates are finite", finite, f"values = {[round(v,3) for v in vals]}")
check("eigenvalue estimates are bounded", bounded)

# ---- Test 4: spectral_norm wrapper alternative (approximate enforcement) ----
print("\nTest 4: power-iteration spectral norm (approximate) converges sigma_1 toward 1")
lin = nn.Linear(64, 64, bias=False)
sn = nn.utils.spectral_norm(lin)
x = torch.randn(128, 64)
# power iteration converges over repeated forward passes (this is by design;
# a single pass gives a loose estimate). Run several, as in real training.
for _ in range(20):
    _ = sn(x)
sigma = torch.linalg.matrix_norm(sn.weight, ord=2).item()
check("spectral_norm converges sigma_1 toward 1", sigma <= 1.1, f"sigma_1 = {sigma:.4f} after 20 iters")

# ---- Summary ----
print("\n" + "=" * 64)
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_total = len(results)
print(f"RESULT: {n_pass}/{n_total} checks passed")
print("=" * 64)
print("\nScope reminder: these are mechanism-correctness unit tests on CPU.")
print("They confirm the math is implemented correctly. They are NOT")
print("production-scale validation, which requires the GPU protocol.")

sys.exit(0 if n_pass == n_total else 1)
