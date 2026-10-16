"""Weight-conditioned policy heads for the v2 RL trainer (design sections 1.3-1.4).

AttentivePolicyHeadV2: one step of the grid walk. Keys come from the 8 candidate cells,
    k_c = W_k [h_c || phi_c]          (phi_c omitted when features are disabled)
and the query summarises the partial line and the weight,
    q = W_q (pool(h, selected) + W_e w + W_s sbar).
Actions are a masked softmax over the feasible candidates. There is no value head.
The module also returns the entropy of the masked distribution.

StartHead: a distribution over all N cells for the start of the line,
    logit_i = (W_ks h_i) . (W_qs (mean_i h_i + W_e w)) / sqrt(d).

The w embedding W_e is owned by the caller so that it can be transferred and frozen
with the heads. The old src/rl/policy.py is left untouched.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _masked_entropy(logits: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Return (log_probs, entropy) for a categorical given by logits; masked entries have zero mass."""
    logp = torch.log_softmax(logits, dim=-1)
    if mask is not None:
        logp = logp.masked_fill(~mask, 0.0)
        p = torch.exp(logp) * mask
    else:
        p = torch.exp(logp)
    entropy = -(p * logp).sum()
    return logp, entropy


class AttentivePolicyHeadV2(nn.Module):
    def __init__(self, embed_dim: int, n_groups: int, feat_dim: int, use_features: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_features = use_features
        self.feat_dim = feat_dim if use_features else 0
        self.key_proj = nn.Linear(embed_dim + self.feat_dim, embed_dim, bias=False)  # W_k
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # W_q
        self.group_proj = nn.Linear(n_groups, embed_dim, bias=False)  # W_s
        self.scale = math.sqrt(embed_dim)

    def forward(
        self,
        cand_h: Tensor,  # [8, d]
        cand_feat: Tensor | None,  # [8, feat_dim] or None
        pooled: Tensor,  # [d] pool of the selected cells
        w_emb: Tensor,  # [d] = W_e w
        sbar: Tensor,  # [m] group satisfaction (cumulative, pct)
        mask: Tensor,  # [8] bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (logits [8], log_probs [8], entropy scalar). Masked logits are -inf."""
        if self.use_features:
            if cand_feat is None:
                raise ValueError("features are enabled for this head but none were given")
            keys_in = torch.cat([cand_h, cand_feat], dim=-1)
        else:
            keys_in = cand_h
        keys = self.key_proj(keys_in)  # [8, d]
        query = self.query_proj(pooled + w_emb + self.group_proj(sbar))  # [d]
        scores = (keys @ query) / self.scale
        logits = scores.masked_fill(~mask, float("-inf"))
        logp, entropy = _masked_entropy(logits, mask)
        return logits, logp, entropy


class StartHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # W_ks
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # W_qs
        self.scale = math.sqrt(embed_dim)

    def forward(self, h: Tensor, w_emb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """h: [N, d] node embeddings; returns (logits [N], log_probs [N], entropy scalar)."""
        keys = self.key_proj(h)  # [N, d]
        query = self.query_proj(h.mean(dim=0) + w_emb)  # [d]
        logits = (keys @ query) / self.scale
        logp, entropy = _masked_entropy(logits, None)
        return logits, logp, entropy
