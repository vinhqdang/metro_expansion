"""Heterogeneous graph state encoder for rail network expansion.

Implements the two-relation message-passing scheme used by MetroGNN (Su et
al., WWW'24 Companion, arXiv:2403.09197, Sec. 3.2), adapted here to encode
only rail-candidate regions -- bus-derived signal enters as node features
(see `src/data`), not as a separate relation type, per the rail-only scope
decision recorded in the design spec.

Node embeddings are updated by aggregating over two edge types separately
and then combining:
    h_s,i^(l) = sum_j e^s_ij * W_s^(l) h_j^(l)   (spatial-contiguity edges)
    h_o,i^(l) = sum_j e^o_ij * W_o^(l) h_j^(l)   (OD-flow edges)
    h_i^(l+1) = tanh(W_c^(l) [h_s,i^(l) || h_o,i^(l)] + h_i^(l))   (residual)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class RelationalMessagePassingLayer(nn.Module):
    """One layer of the two-relation (spatial + OD-flow) message passing scheme."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.spatial_lin = nn.Linear(in_dim, hidden_dim, bias=False)
        self.od_lin = nn.Linear(in_dim, hidden_dim, bias=False)
        self.combine_lin = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        # residual connection requires in_dim == hidden_dim; project if not.
        self.residual_proj = None if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(
        self,
        h: Tensor,  # [num_nodes, in_dim]
        spatial_edge_index: Tensor,  # [2, num_spatial_edges]
        spatial_edge_weight: Tensor,  # [num_spatial_edges]
        od_edge_index: Tensor,  # [2, num_od_edges]
        od_edge_weight: Tensor,  # [num_od_edges]
    ) -> Tensor:
        num_nodes = h.size(0)

        h_spatial = self._aggregate(
            h, spatial_edge_index, spatial_edge_weight, self.spatial_lin, num_nodes
        )
        h_od = self._aggregate(h, od_edge_index, od_edge_weight, self.od_lin, num_nodes)

        combined = self.combine_lin(torch.cat([h_spatial, h_od], dim=-1))
        residual = h if self.residual_proj is None else self.residual_proj(h)
        return torch.tanh(combined + residual)

    @staticmethod
    def _aggregate(
        h: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        lin: nn.Linear,
        num_nodes: int,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, lin.out_features, device=h.device, dtype=h.dtype)
        src, dst = edge_index[0], edge_index[1]
        messages = lin(h[src]) * edge_weight.unsqueeze(-1)
        return scatter(messages, dst, dim=0, dim_size=num_nodes, reduce="sum")


class HeteroGNNEncoder(nn.Module):
    """Stacks `num_layers` RelationalMessagePassingLayer blocks.

    Input node features are raw candidate-region attributes (population
    density, POI density, bus-derived demand-proxy signal, existing rail
    accessibility -- see src/data for feature construction); output is a
    per-node embedding used by the policy/value heads in src/rl/policy.py.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [RelationalMessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_features: Tensor,
        spatial_edge_index: Tensor,
        spatial_edge_weight: Tensor,
        od_edge_index: Tensor,
        od_edge_weight: Tensor,
    ) -> Tensor:
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = layer(h, spatial_edge_index, spatial_edge_weight, od_edge_index, od_edge_weight)
        return h


class FlatEncoder(nn.Module):
    """Non-graph baseline encoder: a linear projection of raw node features
    with no message passing over spatial or OD-flow edges.

    Used as the "flat/tabular state" ablation (manuscript Section 6.3),
    approximating a non-GNN policy such as \\citet{zhang2024}'s. Accepts the
    same edge arguments as HeteroGNNEncoder.forward for interface
    compatibility, but ignores them entirely.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.activation = nn.Tanh()

    def forward(
        self,
        node_features: Tensor,
        spatial_edge_index: Tensor,
        spatial_edge_weight: Tensor,
        od_edge_index: Tensor,
        od_edge_weight: Tensor,
    ) -> Tensor:
        return self.activation(self.input_proj(node_features))
