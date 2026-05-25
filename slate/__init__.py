"""SLATE: Spectral-Lyapunov Annealed Training Estimator.

Stable training for compressed-attention language models. Extends
DeepSeek's Native Sparse Attention (NSA) with three innovations:

  1. Spectrally-constrained compression operator (sigma_1(G) <= 1)
  2. Fisher-information weighted token selection (specified; impl deferred)
  3. Lyapunov-derived annealing schedule with bifurcation guard

UK Patent Application: GB2612127.7 (filed 24 May 2026)
Author: Rene Claudel Mugenzi, Physivitis Ltd
License: MIT
"""

from .spectral import (
    find_compress_linears,
    project_to_unit_ball,
    measure_sigma1,
)
from .lyapunov import (
    stochastic_eigenvalue,
    LyapunovScheduler,
)
from .utils import (
    gaussian_entropy,
    spectral_entropy,
)
from .model import (
    SlateBlock,
    SlateGPT,
)

__version__ = "0.1.0"
__author__ = "Rene Claudel Mugenzi"
__email__ = "rene.mugenzi@physivitis.tech"
__license__ = "MIT"

__all__ = [
    "SlateBlock", "SlateGPT",
    "LyapunovScheduler", "stochastic_eigenvalue",
    "project_to_unit_ball", "measure_sigma1", "find_compress_linears",
    "gaussian_entropy", "spectral_entropy",
]
