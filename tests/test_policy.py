import torch
import pytest

from src.rl.policy import AttentivePolicyValueHead, sample_action


def make_embeddings(num_nodes=6, embed_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, embed_dim, generator=g)


def test_forward_output_shapes():
    embeddings = make_embeddings()
    head = AttentivePolicyValueHead(embed_dim=8)
    query = embeddings.mean(dim=0)
    mask = torch.ones(6, dtype=torch.bool)
    logits, value = head(embeddings, query, mask)
    assert logits.shape == (6,)
    assert value.dim() == 0  # scalar


def test_masked_actions_get_zero_probability():
    embeddings = make_embeddings()
    head = AttentivePolicyValueHead(embed_dim=8)
    query = embeddings.mean(dim=0)
    mask = torch.tensor([True, False, True, False, True, False])
    logits, _ = head(embeddings, query, mask)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs[~mask], torch.zeros(3), atol=1e-6)
    assert probs[mask].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_forward_rejects_non_bool_mask():
    embeddings = make_embeddings()
    head = AttentivePolicyValueHead(embed_dim=8)
    query = embeddings.mean(dim=0)
    bad_mask = torch.ones(6, dtype=torch.float32)
    with pytest.raises(TypeError, match="bool"):
        head(embeddings, query, bad_mask)


def test_forward_rejects_all_infeasible_mask():
    embeddings = make_embeddings()
    head = AttentivePolicyValueHead(embed_dim=8)
    query = embeddings.mean(dim=0)
    mask = torch.zeros(6, dtype=torch.bool)
    with pytest.raises(ValueError, match="no feasible actions"):
        head(embeddings, query, mask)


def test_pool_network_state_uses_selected_nodes_only():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    selected = torch.tensor([True, True, False])
    pooled = AttentivePolicyValueHead.pool_network_state(embeddings, selected)
    assert torch.allclose(pooled, torch.tensor([0.5, 0.5]))


def test_pool_network_state_falls_back_to_all_nodes_when_none_selected():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    selected = torch.zeros(3, dtype=torch.bool)
    pooled = AttentivePolicyValueHead.pool_network_state(embeddings, selected)
    assert torch.allclose(pooled, embeddings.mean(dim=0))


def test_sample_action_only_returns_feasible_actions():
    embeddings = make_embeddings()
    head = AttentivePolicyValueHead(embed_dim=8)
    query = embeddings.mean(dim=0)
    mask = torch.tensor([True, False, True, False, False, False])
    logits, _ = head(embeddings, query, mask)

    seen_actions = set()
    for i in range(200):
        gen = torch.Generator().manual_seed(i)
        action, log_prob = sample_action(logits, generator=gen)
        seen_actions.add(action.item())
        assert torch.isfinite(log_prob)

    assert seen_actions <= {0, 2}


def test_sample_action_gradients_flow_through_log_prob():
    embeddings = make_embeddings()
    embeddings.requires_grad_(True)
    head = AttentivePolicyValueHead(embed_dim=8)
    query = embeddings.mean(dim=0)
    mask = torch.ones(6, dtype=torch.bool)
    logits, value = head(embeddings, query, mask)
    action, log_prob = sample_action(logits)
    loss = -log_prob * 1.0 + value
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
