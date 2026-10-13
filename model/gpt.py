"""nanoGPT-derived transformer with SwiGLU activation.

Designed for torch.compile compatibility and small-model data-constrained
training regimes (100K-token corpora, 10-50M param models).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention.

    Uses F.scaled_dot_product_attention for hardware-accelerated attention
    (flash attention on CUDA, memory-efficient on CPU).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Residual dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # Attention dropout probability (passed to SDPA)
        self.attn_dropout_p = config.dropout

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V in one matmul
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash attention via PyTorch SDPA
        dropout_p = self.attn_dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """SwiGLU MLP block.

    gate = Linear(n_embd -> hidden)
    up   = Linear(n_embd -> hidden)
    down = Linear(hidden -> n_embd)

    output = down(SiLU(gate(x)) * up(x))

    Hidden dim is 8/3 * n_embd rounded up to the nearest multiple of 64
    for hardware alignment.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = 64 * math.ceil((8 * config.n_embd / 3) / 64)

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up, then project down
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Transformer block with pre-norm (LayerNorm before attention and MLP)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT language model.

    Architecture:
        - Token embeddings + learned position embeddings
        - N transformer blocks (pre-norm with SwiGLU MLP)
        - Final LayerNorm
        - Linear head (weight-tied with token embeddings)

    Designed for torch.compile: no dynamic control flow in forward pass,
    fixed tensor shapes within a given (B, T) input.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply residual scaling to output projections
        self._apply_residual_scaling()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal(0, 0.02) for linear/embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # LayerNorm is initialized by PyTorch defaults (weight=1, bias=0)

    def _apply_residual_scaling(self) -> None:
        """Scale residual projections by 1/sqrt(2 * n_layer).

        Applied to the output projection of attention (c_proj) and
        the down projection of SwiGLU MLP. This keeps the residual
        stream variance stable at initialization regardless of depth.
        """
        scale = 1.0 / math.sqrt(2 * self.config.n_layer)
        for block in self.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: Input token indices, shape (B, T) with T <= block_size.
            targets: Target token indices for loss computation, shape (B, T).
                     If None, loss is not computed (inference mode).

        Returns:
            (logits, loss) where logits has shape (B, T, vocab_size) and
            loss is a scalar tensor or None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Position indices (static shape for torch.compile)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Compute logits
        if targets is not None:
            # Training: compute logits for all positions, then loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference: only compute logits for the last position (efficiency)
            logits = self.lm_head(x[:, -1:, :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            idx: Conditioning token indices, shape (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature. 1.0 = no change,
                         <1.0 = more deterministic, >1.0 = more random.
            top_k: If set, only sample from the top-k most likely tokens.

        Returns:
            Token indices including the generated continuation, shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            # Forward pass (inference mode: only last-position logits)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_config(cls, config: ModelConfig) -> GPT:
        """Create a GPT model from a ModelConfig."""
        return cls(config)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters.

        Args:
            non_embedding: If True, subtract position embeddings from the count
                           (token embeddings are weight-tied with lm_head so they
                           are always counted as part of the transformer).

        Returns:
            Parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
