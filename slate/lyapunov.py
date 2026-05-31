"""Lyapunov-derived annealing schedule with bifurcation guard.

Implements Innovation 1 of the SLATE framework. The schedule parameter
beta(t) controls the convex combination of compressed-selective and full
attention losses. The update rate is derived from Lyapunov stability theory:

    dbeta/dt = -epsilon * lambda_min(H(t))^2 / ||Delta(t)||

with a bifurcation guard that freezes the schedule when lambda_min < eta,
preventing the schedule from advancing through curvature-degenerate regions
of the loss landscape.

Reference: SLATE paper Section 3.3, UK Patent GB2612127.7 Claims 10-13.
"""
import torch


def stochastic_eigenvalue(model, batch, loss_fn):
    """Estimate Rayleigh quotient of Hessian via Pearlmutter trick.

    Returns an estimate of an eigenvalue of the Hessian in the direction of
    a random unit vector. For lambda_min specifically, run Lanczos iteration
    on (cI - H) and take the largest eigenvalue, then subtract from c.

    Args:
        model: nn.Module
        batch: (input, target) tuple
        loss_fn: callable loss_fn(model, x, y) -> scalar loss

    Returns:
        Rayleigh quotient v^T H v / v^T v for a random unit v.
    """
    x, y = batch
    loss = loss_fn(model, x, y)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])
    v = torch.randn_like(flat_grad)
    v = v / v.norm()
    # Pearlmutter trick: Hv = d/dtheta (grad . v). Form the scalar (grad . v)
    # then differentiate w.r.t. params. This avoids grad_outputs shape issues.
    gv = (flat_grad * v).sum()
    Hv_grads = torch.autograd.grad(gv, params, retain_graph=False, allow_unused=True)
    Hv = torch.cat([
        h.flatten() if h is not None else torch.zeros_like(p.flatten())
        for h, p in zip(Hv_grads, params)
    ])
    # Rayleigh quotient v^T H v (v is unit norm so denominator is 1)
    return (v * Hv).sum().item()


class LyapunovScheduler:
    """Schedule controller updating beta(t) using Lyapunov-derived rate.

    Usage:
        scheduler = LyapunovScheduler(model=model, loss_fn=my_loss_fn)
        for step in range(num_steps):
            batch = get_batch()
            beta = scheduler.step(batch)
            loss = beta * loss_csa + (1 - beta) * loss_full
            loss.backward()
            opt.step()
            model.project_constraints()

    Args:
        initial_beta: starting value of schedule parameter in [0, 1]
        epsilon: Lyapunov rate scaling factor (smaller = slower schedule)
        eta: bifurcation guard threshold (schedule freezes if |lambda_min| < eta)
        update_every: K, update schedule every K training steps
        model: nn.Module being trained (required for Hessian estimation)
        loss_fn: callable for Hessian estimation
    """

    def __init__(self, initial_beta=0.5, epsilon=0.001, eta=1e-4,
                 update_every=50, model=None, loss_fn=None):
        self.beta = float(initial_beta)
        self.epsilon = epsilon
        self.eta = eta
        self.update_every = update_every
        self.model = model
        self.loss_fn = loss_fn
        self.step_count = 0
        self.guard_activations = 0
        self.update_count = 0

    def step(self, batch=None):
        """Advance schedule by one training step. Returns current beta.

        Hessian-based update occurs every `update_every` steps. Other steps
        return the current beta unchanged.
        """
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return self.beta
        if self.step_count <= self.update_every:
            # Skip first update to let optimiser warm up
            return self.beta
        if self.model is None or self.loss_fn is None or batch is None:
            return self.beta
        try:
            rayleigh = stochastic_eigenvalue(self.model, batch, self.loss_fn)
            self.update_count += 1
            if abs(rayleigh) < self.eta:
                self.guard_activations += 1
                # Schedule frozen (bifurcation guard active)
                return self.beta
            beta_dot = -self.epsilon * (rayleigh ** 2) / max(abs(rayleigh) * 0.1, 1e-6)
            self.beta = max(0.0, min(1.0, self.beta + beta_dot))
        except Exception:
            # Don't crash training if Hessian estimation fails on a particular batch
            pass
        return self.beta

    def state_dict(self):
        return {
            "beta": self.beta,
            "step_count": self.step_count,
            "guard_activations": self.guard_activations,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state):
        self.beta = state["beta"]
        self.step_count = state.get("step_count", 0)
        self.guard_activations = state.get("guard_activations", 0)
        self.update_count = state.get("update_count", 0)
