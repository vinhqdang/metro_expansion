"""Masked attentive actor-critic policy over GNN node embeddings.

Follows the MetroGNN policy design (Su et al., WWW'24 Companion, Fig. 2c):
at each step the agent selects one candidate node (rail extension target) by
scoring all nodes against a query representing the current partial network,
masking out infeasible actions (already-selected nodes, or nodes violating
station-spacing/angle/budget constraints -- mask construction itself lives
in src/data, since it is city/geometry-specific).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentivePolicyValueHead(nn.Module):
    """Given node embeddings and a feasibility mask, produces:
      - action_logits: [num_nodes] unnormalized log-probabilities (masked
        positions set to -inf so they get zero probability after softmax)
      - value: scalar state-value estimate for the actor-critic baseline
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim**0.5

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        node_embeddings: Tensor,  # [num_nodes, embed_dim]
        network_state_embedding: Tensor,  # [embed_dim] -- pooled embedding of selected nodes so far
        feasibility_mask: Tensor,  # [num_nodes] bool, True = feasible action
    ) -> tuple[Tensor, Tensor]:
        if feasibility_mask.dtype != torch.bool:
            raise TypeError("feasibility_mask must be a bool tensor")
        if not feasibility_mask.any():
            raise ValueError("feasibility_mask has no feasible actions -- episode should have terminated")

        query = self.query_proj(network_state_embedding)  # [embed_dim]
        keys = self.key_proj(node_embeddings)  # [num_nodes, embed_dim]
        scores = (keys @ query) / self.scale  # [num_nodes]

        action_logits = scores.masked_fill(~feasibility_mask, float("-inf"))

        # Value estimate is computed from the pooled network-state embedding,
        # not from any single candidate node.
        value = self.value_head(network_state_embedding.unsqueeze(0)).squeeze()

        return action_logits, value

    @staticmethod
    def pool_network_state(node_embeddings: Tensor, selected_mask: Tensor) -> Tensor:
        """Mean-pool the embeddings of already-selected nodes into a single
        vector representing the current partial network. If no nodes are
        selected yet (start of episode), falls back to mean-pooling all
        candidate node embeddings.
        """
        if selected_mask.any():
            return node_embeddings[selected_mask].mean(dim=0)
        return node_embeddings.mean(dim=0)


def sample_action(action_logits: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Sample an action from the masked logits; returns (action_index, log_prob)."""
    dist = torch.distributions.Categorical(logits=action_logits)
    action = dist.sample() if generator is None else _sample_with_generator(dist, generator)
    return action, dist.log_prob(action)


def _sample_with_generator(dist: torch.distributions.Categorical, generator: torch.Generator) -> Tensor:
    # torch.distributions.Categorical.sample() does not accept a generator
    # directly; sample via inverse-CDF using the generator for reproducibility.
    probs = dist.probs
    u = torch.rand(1, generator=generator, device=probs.device)
    cdf = torch.cumsum(probs, dim=-1)
    return torch.searchsorted(cdf, u).clamp(max=probs.numel() - 1).squeeze()
