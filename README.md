# SLATE: Spectral-Lyapunov Annealed Training Estimator

**Provably stable training for compressed-attention language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Patent: GB2612127.7](https://img.shields.io/badge/Patent-GB2612127.7-blue.svg)](https://www.gov.uk/search-for-patent)

SLATE extends [DeepSeek's Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) with three mathematical innovations that give training compressed-attention language models **provable stability guarantees** rather than the empirical stability assumed by current methods.

- **Innovation 1**: Lyapunov-derived annealing schedule with bifurcation guard *(mechanism demonstrated)*
- **Innovation 2**: Fisher-information weighted token selection *(theoretically specified; implementation deferred)*
- **Innovation 3**: Spectrally-constrained compression operator with information-monotonicity guarantee **(fully validated)**

## What SLATE provides that NSA does not

- A compression operator that **cannot amplify errors** (provably 1-Lipschitz)
- A compression operator with **information monotonicity** H(G(X)) ≤ H(X) (empirically verified across 16 layers)
- A training schedule that **cannot pass through curvature-degenerate regions** (Lyapunov-stable with explicit bifurcation guard)

These guarantees matter for AI safety, verified AI, and deployment in regulated contexts (medical, defence, financial, safety-critical) where empirical stability is insufficient and formal verification is required.

## What SLATE does NOT provide

- It does **not** make training **faster** than NSA. SLATE adds guarantees on top of NSA's existing efficiency.
- Innovation 2 (Fisher selection) is **not** currently implemented in working code. It is specified in the paper and patent; validation is deferred.
- All current validation is at **toy scale** (1.85M parameters, Shakespeare, 256-token sequences). Production-scale validation is planned but not done.

## Quick start

```bash
git clone https://github.com/physivitis/slate.git
cd slate
pip install -r requirements.txt
python experiments/run_c3_strict.py     # reproduces validated Innovation 3
python experiments/run_entropy_test.py  # reproduces Patent Claim 17 validation
```

## Validated results (toy scale, A100 GPU, 1500 training steps)

| Condition       | Val Loss | Perplexity | Max σ₁(G)   | Violations | Status |
|-----------------|----------|------------|-------------|------------|--------|
| Full Attention  | 1.9036   | 6.71       | n/a         | n/a        | Baseline |
| NSA Baseline    | 1.6800   | 5.37       | n/a         | n/a        | Baseline |
| SLATE Innov. 3  | 1.6814   | 5.37       | 1.000079    | 0/16       | ✓ PASS |
| SLATE 1+3 combined | 1.7137 | 5.55     | 1.000085    | 0/16       | ✓ PASS |

Entropy test (information monotonicity, Patent Claim 17):

- Gaussian entropy estimator: **0/16** layers violate H(G(X)) ≤ H(X)
- Spectral entropy estimator: **0/16** layers violate H(G(X)) ≤ H(X)

See [paper.md](paper.md) for full technical detail and [experiments/](experiments/) for reproduction scripts.

## Usage

```python
from slate import SlateGPT, LyapunovScheduler

model = SlateGPT(vocab_size=65, seq_len=256).cuda()
model.project_constraints()  # initial spectral projection

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(num_steps):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.project_constraints()  # enforce σ₁(G) ≤ 1 after each step
```

## Citation

If you use SLATE, please cite:

```bibtex
@software{mugenzi_slate_2026,
  author       = {Mugenzi, Rene Claudel},
  title        = {SLATE: Spectral-Lyapunov Annealed Training Estimator},
  year         = {2026},
  publisher    = {Physivitis Ltd},
  url          = {https://github.com/physivitis/slate},
  note         = {UK Patent Application GB2612127.7}
}
```

## Patent status

The SLATE framework is the subject of UK Patent Application **GB2612127.7**, by Physivitis Ltd, inventor Rene Claudel Mugenzi.

This code is released under the MIT licence for research and commercial use. The patent grants Physivitis Ltd defensive priority over the underlying methods.

## Author and affiliation

**Rene Claudel Mugenzi**  
Founder and CEO, Physivitis Ltd  
rene.mugenzi@physivitis.tech  
https://physivitis.tech

## Acknowledgements

SLATE builds directly on the Native Sparse Attention method of Yuan et al. (DeepSeek-AI), ACL 2025. The reference implementation used in this work is from Phil Wang ([lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch)), MIT licensed.

## License

[MIT License](LICENSE) — © 2026 Physivitis Ltd
