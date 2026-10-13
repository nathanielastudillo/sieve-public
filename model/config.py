"""Model configuration for SIEVE transformer models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a nanoGPT-style transformer model.

    Defaults are tuned for the ~20M param regime on 100K-token corpora,
    following Slowrun findings for data-constrained training.
    """

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512
    dropout: float = 0.1
    bias: bool = False
    vocab_size: int = 50257

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def estimate_params(self) -> dict[str, int]:
        """Estimate parameter counts by component.

        Returns a dict with per-component counts and a 'total' key.
        SwiGLU MLP uses 3 projection matrices of size (n_embd, 8/3 * n_embd)
        rounded to the nearest multiple of 64 for hardware efficiency.
        """
        E = self.n_embd
        V = self.vocab_size
        T = self.block_size
        L = self.n_layer

        # SwiGLU hidden dim: 8/3 * n_embd, rounded up to multiple of 64
        mlp_hidden = 64 * math.ceil((8 * E / 3) / 64)

        # Token embeddings
        tok_emb = V * E
        # Position embeddings
        pos_emb = T * E

        # Per-block params
        # Attention: Q, K, V projections + output projection
        attn_per_block = 4 * E * E  # qkv combined + out
        # LayerNorm for attention (pre-norm)
        ln_attn_per_block = 2 * E if self.bias else E
        # SwiGLU MLP: gate (E -> H), up (E -> H), down (H -> E)
        mlp_per_block = 3 * E * mlp_hidden
        # LayerNorm for MLP (pre-norm)
        ln_mlp_per_block = 2 * E if self.bias else E

        block_params = attn_per_block + ln_attn_per_block + mlp_per_block + ln_mlp_per_block
        transformer_blocks = L * block_params

        # Final layer norm
        final_ln = 2 * E if self.bias else E

        # lm_head is weight-tied with token embeddings, so not counted separately
        # If bias=True on attention/MLP, add bias terms
        attn_bias = 0
        mlp_bias = 0
        if self.bias:
            attn_bias = L * (3 * E + E)  # qkv bias + out bias
            mlp_bias = L * (2 * mlp_hidden + E)  # gate bias + up bias + down bias

        total = tok_emb + pos_emb + transformer_blocks + final_ln + attn_bias + mlp_bias

        return {
            "token_embeddings": tok_emb,
            "position_embeddings": pos_emb,
            "transformer_blocks": transformer_blocks,
            "final_layernorm": final_ln,
            "attention_bias": attn_bias,
            "mlp_bias": mlp_bias,
            "total": total,
        }

    @property
    def n_params(self) -> int:
        """Total estimated parameter count."""
        return self.estimate_params()["total"]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON/YAML storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Deserialize from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def __repr__(self) -> str:
        params = self.n_params
        if params >= 1_000_000:
            human = f"{params / 1_000_000:.1f}M"
        elif params >= 1_000:
            human = f"{params / 1_000:.1f}K"
        else:
            human = str(params)
        return (
            f"ModelConfig(n_layer={self.n_layer}, n_head={self.n_head}, "
            f"n_embd={self.n_embd}, block_size={self.block_size}, "
            f"dropout={self.dropout}, bias={self.bias}, "
            f"vocab_size={self.vocab_size}, ~{human} params)"
        )
