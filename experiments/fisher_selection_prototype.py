"""
Fisher-selection stability prototype (CPU, toy scale).

PURPOSE: De-risk the single hardest part of the production validation
protocol (Experiment 2). The earlier toy attempt at Innovation 2 exploded
(gradient norm 476.93) because it applied Fisher weighting as a multiplicative
gate perturbation rather than as a top-k SELECTION criterion as the patent
specifies.

This prototype isolates the selection mechanism from the full NSA backbone
and answers ONE question: can Fisher-weighted top-k block selection be made
numerically stable?

WHAT THIS IS:  a mechanism-correctness and stability check on CPU at toy size.
WHAT THIS IS NOT:  production-scale validation. Any apparent quality difference
at this scale is NOT meaningful evidence. That requires the GPU runs in the
protocol (Pythia-160M, billions of tokens, 5 seeds).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# ----------------------------------------------------------------------
# A minimal "compressed-attention" layer with explicit block selection.
# Queries attend to a set of compressed K/V blocks; only top-k blocks
# per query are used. Two selection rules are provided.
# ----------------------------------------------------------------------

class BlockSelectAttention(nn.Module):
    def __init__(self, d_model, n_blocks, k_select, rule="cosine"):
        super().__init__()
        self.d = d_model
        self.n_blocks = n_blocks
        self.k = k_select
        self.rule = rule
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q_tokens, kv_blocks, optimizer=None):
        # q_tokens: (B, T, d)   kv_blocks: (B, N, d)
        B, T, d = q_tokens.shape
        N = kv_blocks.shape[1]
        q = self.q_proj(q_tokens)                 # (B,T,d)
        k = self.k_proj(kv_blocks)                # (B,N,d)
        v = self.v_proj(kv_blocks)                # (B,N,d)

        # raw attention scores query->block
        scores = torch.einsum("btd,bnd->btn", q, k) / math.sqrt(d)  # (B,T,N)

        if self.rule == "cosine":
            sel_score = scores
        elif self.rule == "fisher":
            sel_score = self._fisher_utility(scores, q, k, optimizer)
        else:
            raise ValueError(self.rule)

        # top-k block selection (DISCRETE; gradients do NOT flow through the
        # choice of which blocks, only through the attention over selected ones.
        # This stop-gradient on the index is the key to numerical stability.)
        topk_idx = sel_score.topk(self.k, dim=-1).indices       # (B,T,k)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)

        # masked softmax over selected blocks (mask is detached: selection is
        # a routing decision, not a differentiable weight)
        masked = scores.masked_fill(mask.detach() == 0, float("-inf"))
        attn = masked.softmax(dim=-1)                            # (B,T,N)
        out = torch.einsum("btn,bnd->btd", attn, v)
        return self.out(out)

    def _fisher_utility(self, scores, q, k, optimizer):
        """Fisher-weighted utility for block selection.

        phi_ialpha ~ (score_ialpha)^2 / F_diag, where F_diag is read from the
        Adam second-moment state (v_t) of the projection weights, used as a
        diagonal Fisher proxy. Reused at zero extra cost, per the patent.

        STABILITY MEASURES (all of these matter):
          - Fisher diagonal is FLOORED away from zero (prevents 1/~0 blow-up).
          - Utility is computed under no_grad and DETACHED (selection is a
            routing signal, not a backprop path -> no gradient explosion).
          - Per-query normalisation keeps the utility on a sane scale.
        """
        with torch.no_grad():
            # Read Adam's v_t for the k_proj weight as a diagonal Fisher proxy.
            fisher_scalar = 1.0
            if optimizer is not None:
                st = optimizer.state.get(self.k_proj.weight, {})
                if "exp_avg_sq" in st:
                    # mean second moment -> scalar Fisher proxy, floored
                    fisher_scalar = st["exp_avg_sq"].mean().clamp_min(1e-8).item()
            # utility: squared selection score scaled by inverse Fisher proxy
            phi = (scores ** 2) / (fisher_scalar + 1e-8)
            # per-query normalisation (numerical hygiene)
            phi = phi / (phi.amax(dim=-1, keepdim=True) + 1e-8)
        return phi  # detached: used only to pick indices


# ----------------------------------------------------------------------
# Synthetic retrieval task: the loss depends on a specific "signal" block
# that is NOT necessarily the highest cosine-similarity block. This is the
# regime where Fisher selection could in principle differ from cosine.
# (We are testing STABILITY here, not claiming advantage.)
# ----------------------------------------------------------------------

def make_task(B, T, N, d):
    q = torch.randn(B, T, d)
    blocks = torch.randn(B, N, d)
    # target depends on block index (T % N) per position -> the "informative"
    # block, deliberately decorrelated from raw similarity
    target_block = torch.arange(T) % N
    tgt = blocks[:, target_block, :]                 # (B,T,d)
    return q, blocks, tgt


def run(rule, steps=400, lr=1e-3):
    d, B, T, N, k = 32, 8, 16, 12, 4
    model = BlockSelectAttention(d, N, k, rule=rule)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    q, blocks, tgt = make_task(B, T, N, d)

    max_grad = 0.0
    nan_steps = 0
    losses = []
    for step in range(steps):
        out = model(q, blocks, optimizer=opt)
        loss = F.mse_loss(out, tgt)
        if torch.isnan(loss):
            nan_steps += 1
            break
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()  # measure, don't clip hard
        max_grad = max(max_grad, gn)
        opt.step()
        losses.append(loss.item())
    return {
        "rule": rule,
        "final_loss": losses[-1] if losses else float("nan"),
        "max_grad_norm": max_grad,
        "nan_steps": nan_steps,
        "completed": len(losses),
    }


print("=" * 64)
print("FISHER-SELECTION STABILITY PROTOTYPE (CPU, toy scale)")
print("De-risking Experiment 2 of the production validation protocol")
print("=" * 64)

for rule in ["cosine", "fisher"]:
    r = run(rule)
    print(f"\n[{rule.upper()} selection]")
    print(f"  steps completed : {r['completed']}/400")
    print(f"  NaN occurrences : {r['nan_steps']}")
    print(f"  max grad norm   : {r['max_grad_norm']:.3f}")
    print(f"  final loss      : {r['final_loss']:.4f}")
    stable = (r['nan_steps'] == 0 and r['max_grad_norm'] < 50)
    print(f"  STABLE          : {'YES' if stable else 'NO'}")

print("\n" + "=" * 64)
print("INTERPRETATION (read carefully):")
print("- This tests whether Fisher-weighted top-k SELECTION runs stably.")
print("- The earlier toy attempt exploded (grad 476.93) using a multiplicative")
print("  gate; this version uses detached top-k selection + floored Fisher.")
print("- A stable run here de-risks the NSA fork. It does NOT prove advantage.")
print("- Advantage requires the GPU protocol: Pythia-160M, billions of tokens.")
print("=" * 64)
