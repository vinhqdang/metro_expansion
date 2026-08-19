import torch

from src.gnn.encoder import FlatEncoder, HeteroGNNEncoder, RelationalMessagePassingLayer


def make_toy_graph(num_nodes=6, in_dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    node_features = torch.randn(num_nodes, in_dim, generator=g)

    # a small ring of spatial-contiguity edges (undirected, both directions)
    spatial_src = torch.arange(num_nodes)
    spatial_dst = (spatial_src + 1) % num_nodes
    spatial_edge_index = torch.stack(
        [torch.cat([spatial_src, spatial_dst]), torch.cat([spatial_dst, spatial_src])]
    )
    spatial_edge_weight = torch.ones(spatial_edge_index.size(1))

    # a few random OD-flow edges with positive weights
    od_edge_index = torch.randint(0, num_nodes, (2, 8), generator=g)
    od_edge_weight = torch.rand(8, generator=g) + 0.1

    return node_features, spatial_edge_index, spatial_edge_weight, od_edge_index, od_edge_weight


def test_relational_layer_output_shape():
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=8)
    layer = RelationalMessagePassingLayer(in_dim=8, hidden_dim=8)
    out = layer(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()


def test_relational_layer_projects_residual_when_dims_differ():
    # Exercises the residual_proj branch (in_dim != hidden_dim), used
    # internally by HeteroGNNEncoder's first layer after input_proj but not
    # otherwise reachable through HeteroGNNEncoder itself.
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=4)
    layer = RelationalMessagePassingLayer(in_dim=4, hidden_dim=10)
    assert layer.residual_proj is not None
    out = layer(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
    assert out.shape == (6, 10)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert layer.residual_proj.weight.grad is not None
    assert torch.isfinite(layer.residual_proj.weight.grad).all()


def test_relational_layer_handles_empty_edge_set():
    node_features, spatial_ei, spatial_ew, _, _ = make_toy_graph(in_dim=8)
    empty_ei = torch.empty((2, 0), dtype=torch.long)
    empty_ew = torch.empty((0,))
    layer = RelationalMessagePassingLayer(in_dim=8, hidden_dim=8)
    out = layer(node_features, spatial_ei, spatial_ew, empty_ei, empty_ew)
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()


def test_encoder_output_shape_and_layer_count():
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=4)
    encoder = HeteroGNNEncoder(in_dim=4, hidden_dim=16, num_layers=3)
    assert len(encoder.layers) == 3
    out = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
    assert out.shape == (6, 16)
    assert torch.isfinite(out).all()


def test_encoder_rejects_zero_layers():
    import pytest

    with pytest.raises(ValueError, match="num_layers"):
        HeteroGNNEncoder(in_dim=4, hidden_dim=16, num_layers=0)


def test_encoder_gradients_flow_to_input_projection():
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=4)
    encoder = HeteroGNNEncoder(in_dim=4, hidden_dim=8, num_layers=2)
    out = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
    loss = out.sum()
    loss.backward()
    assert encoder.input_proj.weight.grad is not None
    assert torch.isfinite(encoder.input_proj.weight.grad).all()
    assert encoder.input_proj.weight.grad.abs().sum() > 0


def test_encoder_runs_on_cuda_if_available():
    if not torch.cuda.is_available():
        return
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=4)
    device = torch.device("cuda")
    encoder = HeteroGNNEncoder(in_dim=4, hidden_dim=8, num_layers=2).to(device)
    out = encoder(
        node_features.to(device),
        spatial_ei.to(device),
        spatial_ew.to(device),
        od_ei.to(device),
        od_ew.to(device),
    )
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()


def test_flat_encoder_ignores_edges_entirely():
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=4)
    encoder = FlatEncoder(in_dim=4, hidden_dim=8)
    out_with_edges = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)

    empty_ei = torch.empty((2, 0), dtype=torch.long)
    empty_ew = torch.empty((0,))
    out_without_edges = encoder(node_features, empty_ei, empty_ew, empty_ei, empty_ew)

    assert out_with_edges.shape == (6, 8)
    assert torch.allclose(out_with_edges, out_without_edges)


def test_flat_encoder_gradients_flow():
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = make_toy_graph(in_dim=4)
    encoder = FlatEncoder(in_dim=4, hidden_dim=8)
    out = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
    out.sum().backward()
    assert encoder.input_proj.weight.grad is not None
    assert torch.isfinite(encoder.input_proj.weight.grad).all()
