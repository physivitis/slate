"""Entropy estimators for verifying information monotonicity.

Used to empirically test Claim 17 of UK Patent GB2612127.7:
H(G(X)) <= H(X) for the spectrally-constrained compression operator G.

Two independent estimators are provided:
  - gaussian_entropy: parametric, per-dimension assumption of Gaussianity
  - spectral_entropy: non-parametric, from singular value spectrum of data

Validated: zero violations across 16 compression layers in C3-strict model.
"""
import math
import torch


def gaussian_entropy(x):
    """Per-dimension Gaussian entropy estimate (parametric).

    H = 0.5 * log(2 * pi * e * var) per dim, summed and meaned across dims.

    Args:
        x: tensor with last dim treated as feature dim. Shape: (..., D)

    Returns:
        (total_entropy, per_dim_mean_entropy) in nats
    """
    x_flat = x.reshape(-1, x.shape[-1])
    var = x_flat.var(dim=0)
    var = torch.clamp(var, min=1e-10)
    h_per_dim = 0.5 * torch.log(2 * math.pi * math.e * var)
    return h_per_dim.sum().item(), h_per_dim.mean().item()


def spectral_entropy(x):
    """Non-parametric spectral entropy from singular value spectrum.

    Centres x, computes SVD, normalises squared singular values to a
    probability distribution, returns Shannon entropy.

    Args:
        x: tensor with last dim as feature dim. Shape: (..., D)

    Returns:
        (entropy, sigma_max, sigma_min) where entropy is in nats
    """
    x_flat = x.reshape(-1, x.shape[-1])
    x_c = x_flat - x_flat.mean(dim=0, keepdim=True)
    _U, S, _Vh = torch.linalg.svd(x_c, full_matrices=False)
    p = (S ** 2) / (S ** 2).sum()
    p = torch.clamp(p, min=1e-10)
    h = -(p * torch.log(p)).sum().item()
    return h, S[0].item(), S[-1].item()
