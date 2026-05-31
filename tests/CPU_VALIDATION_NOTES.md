# CPU Validation Notes (what these scripts do and do not establish)

This directory contains two CPU-runnable scripts produced to support the
production validation effort. **Neither is production-scale validation.**
Read this file before citing any result from them.

## tests/test_correctness.py — mechanism-correctness harness

Confirms the SLATE package implements its mathematical claims correctly.
All checks are scale-independent linear-algebra facts, runnable on any CPU.

Result: **8/8 checks pass.**

| Check | What it confirms |
|-------|------------------|
| SVD projection enforces sigma_1 <= 1 | After projection, max sigma_1 = 1.000001 across 8 inflated layers |
| Constraint holds on all layers | 0 violations / 8 |
| Gaussian entropy monotonicity | 0 violations / 16 layers (H(G(X)) <= H(X)) |
| Spectral entropy monotonicity | 0 violations / 16 layers |
| Eigenvalue estimator finite | Pearlmutter HVP returns finite values |
| Eigenvalue estimator bounded | values stay in sane range |
| spectral_norm converges | sigma_1 = 1.004 after 20 power iterations |

**This harness caught a real bug** in the original `lyapunov.py`: the
Hessian-vector product used an incorrect `grad_outputs` shape and crashed.
Fixed to the standard Pearlmutter scalar formulation. This is exactly what a
correctness harness is for.

**Scope:** these are unit tests. They prove the math is implemented correctly.
They prove nothing about production scale, quality, or advantage.

## experiments/fisher_selection_prototype.py — Experiment 2 de-risking

The single hardest part of the production validation protocol is the Fisher
selection (Experiment 2). The earlier toy attempt exploded (gradient norm
476.93) because it applied Fisher weighting as a multiplicative gate rather
than as a top-k selection criterion.

This prototype isolates the selection mechanism and answers one question:
**can Fisher-weighted top-k selection be made numerically stable?**

Result: **Yes.**

| Selection rule | Max grad norm | NaN steps | Stable |
|----------------|---------------|-----------|--------|
| Cosine (baseline) | 0.135 | 0 | YES |
| Fisher (Innovation 2) | 0.143 | 0 | YES |

The stability fixes that matter:
- top-k selection indices are **detached** (selection is a routing decision,
  not a differentiable weight) — this is what prevents the gradient explosion
- the Fisher diagonal is **floored** away from zero (prevents 1/~0 blow-up)
- per-query **normalisation** of the utility keeps it on a sane scale

**Scope:** this de-risks the NSA fork. It does NOT show Fisher selection is
better than cosine. At this toy scale Fisher actually did slightly worse
(final loss 0.534 vs 0.405), which is meaningless — advantage requires the
GPU protocol (Pythia-160M, billions of tokens, 5 seeds, tokens-to-target
measurement). The point here is only: the mechanism no longer explodes.

## What still requires the GPU protocol

Everything that matters commercially and scientifically:
- Quality preservation at 410M / 8K context (Experiment 1)
- Fisher selection sample-efficiency advantage (Experiment 2)
- Lyapunov schedule spike reduction under stress (Experiment 3)

See SLATE_Production_Validation_Protocol for the full design, pre-registered
pass/fail criteria, and costed compute budget.
