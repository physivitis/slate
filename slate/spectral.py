"""Spectral constraint enforcement for compression operators.

Implements Innovation 3 of the SLATE framework: enforcing sigma_1(G) <= 1
on the learned compression operator G, providing 1-Lipschitz behaviour
and information monotonicity by the data processing inequality.

Two enforcement modes:
  - project_to_unit_ball: exact enforcement via SVD projection (post-step).
    Validated in C3-strict with max sigma_1 = 1.000079 across 16 layers.
  - (spectral_norm wrapper from torch.nn.utils): approximate enforcement
    via power iteration. Use sparingly; conflicts with explicit weight assignment.

Reference: SLATE paper Section 3.1, UK Patent GB2612127.7 Claims 2-5.
"""
import torch
import torch.nn as nn


def find_compress_linears(module, target=("compress",), exclude=("combine",)):
    """Find Linear sub-modules that act as compression operators.

    In the lucidrains NSA implementation, the relevant layers have names
    containing 'compress' (e.g., 'k_compress', 'v_compress') but not 'combine'.

    Args:
        module: parent module (typically a SparseAttention block).
        target: tuple of substrings, layer name must contain at least one.
        exclude: tuple of substrings, layer name must contain none.

    Returns:
        List of (name, layer) tuples.
    """
    found = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            lname = name.lower()
            if any(t in lname for t in target) and not any(e in lname for e in exclude):
                found.append((name, m))
    return found


def project_to_unit_ball(layers, target_max=1.0):
    """Exact enforcement: clamp all singular values to [0, target_max] via SVD.

    Apply AFTER each optimizer.step() call. Modifies layer weights in place.

    Args:
        layers: list of (name, nn.Linear) tuples to constrain.
        target_max: upper bound on sigma_1 (default 1.0 for unit ball).
    """
    with torch.no_grad():
        for _name, m in layers:
            W = m.weight
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            if S[0] > target_max:
                S_clamped = torch.clamp(S, max=target_max)
                m.weight.copy_(U @ torch.diag(S_clamped) @ Vh)


def measure_sigma1(layers):
    """Measure the largest singular value of each constrained layer.

    Useful for verifying the constraint is being enforced correctly.

    Args:
        layers: list of (name, nn.Linear) tuples.

    Returns:
        List of (name, sigma_1) tuples.
    """
    return [
        (name, torch.linalg.matrix_norm(m.weight, ord=2).item())
        for name, m in layers
    ]
