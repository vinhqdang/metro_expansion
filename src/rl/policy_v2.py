"""Weight-conditioned policy heads for the v2 RL trainer (design sections 1.3-1.4).

AttentivePolicyHeadV2: one step of the grid walk. Keys come from the 8 candidate cells,
    k_c = W_k [h_c || phi_c]          (phi_c omitted when features are disabled)
and the query summarises the partial line and the weight,
    q = W_q (pool(h, selected) + W_e w + W_s sbar).
Actions are a masked softmax over the feasible candidates. There is no value head.
The module also returns the entropy of the masked distribution.

StartHead: a distribution over all N cells for the start of the line,
    logit_i = (W_ks h_i) . (W_qs (mean_i h_i + W_e w)) / sqrt(d).

Scalarised prior (optional, prior_lam > 0 / prior_lam_s > 0): both heads add a fixed,
non-learned term lam * sum_j w_j z_j, where z_j is a standardised per-objective potential
(moves: one-step gains of D, GGI and A over the feasible candidates; start: static per-cell
OD degree, group-balance and uncovered-mass potentials). The learned attention is a residual
on top of this w-scalarised heuristic, so the dependence on w cannot be trained away.

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
    def __init__(self, embed_dim: int, n_groups: int, feat_dim: int, use_features: bool = True,
                 prior_lam: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_features = use_features
        self.prior_lam = float(prior_lam)
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
        prior: Tensor | None = None,  # [8, 3] standardised per-objective gains (see module doc)
        w: Tensor | None = None,  # [3] raw weight on the simplex
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
        if self.prior_lam > 0.0 and prior is not None:
            scores = scores + self.prior_lam * (prior @ w)
        logits = scores.masked_fill(~mask, float("-inf"))
        logp, entropy = _masked_entropy(logits, mask)
        return logits, logp, entropy


class StartHead(nn.Module):
    def __init__(self, embed_dim: int, prior_lam: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.prior_lam = float(prior_lam)
        # [N, 3] standardised static cell potentials; city-specific, so not saved in checkpoints
        self.register_buffer("psi", torch.zeros(0, 3), persistent=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # W_ks
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # W_qs
        self.scale = math.sqrt(embed_dim)

    def forward(self, h: Tensor, w_emb: Tensor, w: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """h: [N, d] node embeddings; returns (logits [N], log_probs [N], entropy scalar)."""
        keys = self.key_proj(h)  # [N, d]
        query = self.query_proj(h.mean(dim=0) + w_emb)  # [d]
        logits = (keys @ query) / self.scale
        if self.prior_lam > 0.0 and w is not None and self.psi.shape[0] == logits.shape[0]:
            logits = logits + self.prior_lam * (self.psi @ w)
        logp, entropy = _masked_entropy(logits, None)
        return logits, logp, entropy
