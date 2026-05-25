"""SLATE model: NSA backbone with spectrally-constrained compression.

Wraps the lucidrains Native Sparse Attention implementation, adds spectral
constraint enforcement on the compression layers, and provides hooks for
the Lyapunov scheduler.

Typical usage:
    from slate import SlateGPT, LyapunovScheduler

    model = SlateGPT(vocab_size=65, seq_len=256).cuda()
    model.project_constraints()  # initial projection

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(num_steps):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.project_constraints()  # enforce sigma_1(G) <= 1 after each step

Reference: SLATE paper Section 3, UK Patent GB2612127.7.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from native_sparse_attention_pytorch import SparseAttention
    _NSA_AVAILABLE = True
except ImportError:
    _NSA_AVAILABLE = False

from .spectral import find_compress_linears, project_to_unit_ball, measure_sigma1


class SlateBlock(nn.Module):
    """Transformer block using NSA with spectrally-constrained compression."""

    def __init__(self, d_model=192, n_head=6, d_head=32, dropout=0.1,
                 sliding_window=16, compress_block=8, compress_stride=4,
                 selection_block=8, num_selected=4):
        super().__init__()
        if not _NSA_AVAILABLE:
            raise ImportError(
                "native_sparse_attention_pytorch is required. "
                "Install with: pip install native-sparse-attention-pytorch"
            )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = SparseAttention(
            dim=d_model, dim_head=d_head, heads=n_head,
            sliding_window_size=sliding_window,
            compress_block_size=compress_block,
            compress_block_sliding_stride=compress_stride,
            selection_block_size=selection_block,
            num_selected_blocks=num_selected,
            causal=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SlateGPT(nn.Module):
    """GPT-style model trained with SLATE.

    Combines: token + positional embeddings, stack of SlateBlocks, output
    head. The constrained_layers attribute lists all compression layers
    that should be spectrally projected after each optimizer step.

    Args:
        vocab_size: vocabulary size
        seq_len: maximum sequence length
        d_model: hidden dimension
        n_head: number of attention heads
        d_head: dimension per head
        n_layer: number of transformer blocks
        dropout: dropout probability for FFN
    """

    def __init__(self, vocab_size, seq_len=256, d_model=192, n_head=6,
                 d_head=32, n_layer=4, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            SlateBlock(d_model=d_model, n_head=n_head, d_head=d_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Collect all compression-layer references for spectral projection
        self.constrained_layers = []
        for i, blk in enumerate(self.blocks):
            found = find_compress_linears(blk.attn)
            self.constrained_layers.extend(
                [(f"block{i}.{n}", m) for n, m in found]
            )

    def forward(self, idx, targets=None):
        _B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def project_constraints(self, target_max=1.0):
        """Enforce sigma_1(G) <= target_max on all constrained layers.

        Call after each optimizer.step() to maintain the spectral constraint.
        """
        project_to_unit_ball(self.constrained_layers, target_max=target_max)

    def measure_constraints(self):
        """Return current max singular values for all constrained layers."""
        return measure_sigma1(self.constrained_layers)

    def max_sigma(self):
        """Return the maximum sigma_1 across all constrained layers."""
        sigmas = self.measure_constraints()
        return max(s for _, s in sigmas) if sigmas else 0.0
