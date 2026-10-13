"""Muon optimizer — momentum in the orthogonalized gradient space.

From the Slowrun/nanoGPT lineage. Key insight: for 2D weight matrices,
orthogonalize the gradient via Newton-Schulz iteration before applying
momentum. This produces approximate steepest descent under the spectral
norm, which is a better geometry for neural network weight matrices than
the Frobenius norm used by SGD/Adam.

Non-2D parameters (embeddings, layer norms, biases) fall back to AdamW.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


def _newton_schulz_orthogonalize(
    G: torch.Tensor, steps: int = 5
) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz iteration.

    Computes an approximation to U from the polar decomposition G = U @ S,
    where U is orthogonal (or semi-orthogonal for non-square matrices).

    The iteration is:
        X_{k+1} = (3 * X_k - X_k @ X_k^T @ X_k) / 2

    which converges quadratically when the singular values of X_0 are in (0, sqrt(3)).
    We ensure this by scaling G so its spectral norm is ~1.

    For tall matrices (rows > cols), we operate on G^T and transpose back
    to maintain efficiency.

    Args:
        G: Gradient matrix, shape (fan_out, fan_in).
        steps: Number of Newton-Schulz iterations.

    Returns:
        Orthogonalized gradient, same shape as G.
    """
    assert G.ndim == 2

    # Work with the smaller dimension on the outside for efficiency
    transposed = False
    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True

    # Scale so all singular values are safely in the NS convergence basin (0, sqrt(3)).
    # ||G||_F >= ||G||_2 always, so dividing by Frobenius norm guarantees max
    # singular value <= 1, well within the convergence radius.
    # (The previous Frobenius/sqrt(m) estimate diverged for low-rank/concentrated
    # gradients like attention layers.)
    scale = G.norm()  # Frobenius norm
    if scale <= 0:
        return G  # zero gradient, nothing to do
    X = G / scale

    # Newton-Schulz iterations: X <- (3X - X @ X^T @ X) / 2
    # Converges quadratically when singular values of X are in (0, sqrt(3)).
    for _ in range(steps):
        A = X @ X.T
        X = (3.0 * X - A @ X) / 2.0

    if transposed:
        X = X.T

    return X


class Muon(Optimizer):
    """Muon optimizer: momentum in the Newton-Schulz orthogonalized gradient space.

    For 2D weight matrices (attention projections, MLP layers), the gradient
    is orthogonalized via Newton-Schulz iteration before momentum is applied.
    This implements approximate steepest descent under the spectral norm.

    For non-2D parameters (embeddings, biases, layer norms), AdamW is used
    as a fallback. These parameters should be placed in a separate param group
    with 'muon': False.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz orthogonalization steps (default: 5).
        adamw_lr: Learning rate for AdamW fallback params (default: 3e-4).
        adamw_betas: Betas for AdamW fallback (default: (0.9, 0.95)).
        adamw_eps: Epsilon for AdamW fallback (default: 1e-8).
        adamw_wd: Weight decay for AdamW fallback (default: 0.1).

    Example:
        # Typical usage with a GPT model:
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if p.ndim == 2 and 'wte' not in name and 'wpe' not in name and 'ln' not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        optimizer = Muon([
            {'params': muon_params},
            {'params': adamw_params, 'muon': False},
        ], lr=0.02)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.1,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            muon=True,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("muon", True)

            if use_muon:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        """Muon update for 2D weight matrices."""
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad

            # For non-2D params that ended up in a muon group, fall back to SGD
            if grad.ndim != 2:
                p.add_(grad, alpha=-lr)
                continue

            # Newton-Schulz orthogonalization
            ortho_grad = _newton_schulz_orthogonalize(grad.float(), steps=ns_steps)
            ortho_grad = ortho_grad.to(dtype=p.dtype)

            # Momentum
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(ortho_grad)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(ortho_grad)

            if nesterov:
                # Nesterov: use grad + momentum * buffer
                update = ortho_grad + momentum * buf
            else:
                update = buf

            # Apply update (scale by sqrt of fan_in * fan_out) for dimension-normalized step
            # This ensures the step size is independent of parameter matrix dimensions
            scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        """AdamW update for non-2D parameters (embeddings, biases, norms)."""
        lr = group.get("adamw_lr", group["lr"])
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["adamw_wd"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.float()
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Decoupled weight decay
            p.mul_(1.0 - lr * wd)

            # Adam moment updates
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            # Bias correction
            step = state["step"]
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step

            corrected_avg = exp_avg / bc1
            corrected_avg_sq = exp_avg_sq / bc2

            # Update
            denom = corrected_avg_sq.sqrt().add_(eps)
            p.add_(corrected_avg / denom, alpha=-lr)


def configure_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.1,
    nesterov: bool = True,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
) -> Muon:
    """Configure a Muon optimizer for a GPT model.

    Automatically separates parameters into:
    - 2D weight matrices (attention, MLP) -> Muon updates
    - Everything else (embeddings, layer norms, biases) -> AdamW updates

    Embedding and lm_head weights always use AdamW because they are
    lookup tables / output projections with different geometry than
    internal weight matrices.

    Args:
        model: The GPT model.
        muon_lr: Learning rate for Muon-updated parameters.
        adamw_lr: Learning rate for AdamW-updated parameters.
        momentum: Momentum for Muon.
        weight_decay: Weight decay for AdamW parameters.
        nesterov: Whether to use Nesterov momentum in Muon.
        ns_steps: Number of Newton-Schulz iterations.
        betas: AdamW beta coefficients.

    Returns:
        Configured Muon optimizer instance.
    """
    muon_params = []
    adamw_params = []

    # Names that should use AdamW regardless of shape
    adamw_names = {"wte", "wpe", "ln_1", "ln_2", "ln_f", "lm_head"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if any adamw_name appears in the parameter name
        use_adamw = any(n in name for n in adamw_names)

        if param.ndim == 2 and not use_adamw:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    param_groups = [
        {"params": muon_params, "muon": True},
        {"params": adamw_params, "muon": False},
    ]

    return Muon(
        param_groups,
        lr=muon_lr,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adamw_lr=adamw_lr,
        adamw_betas=betas,
        adamw_eps=1e-8,
        adamw_wd=weight_decay,
    )
