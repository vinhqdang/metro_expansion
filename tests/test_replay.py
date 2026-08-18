import numpy as np
import pytest

from src.rl.replay import PrioritizedReplayBuffer, Transition


def make_transition(i: int) -> Transition:
    return Transition(state=i, action=0, reward=float(i), next_state=i + 1, done=False)


def test_add_and_len():
    buf = PrioritizedReplayBuffer(capacity=5)
    assert len(buf) == 0
    for i in range(3):
        buf.add(make_transition(i))
    assert len(buf) == 3


def test_capacity_wraps_around():
    buf = PrioritizedReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(make_transition(i))
    assert len(buf) == 3  # never exceeds capacity
    # oldest transitions (0, 1) should have been overwritten by 3, 4
    states = {t.state for t in buf._transitions if t is not None}
    assert states == {2, 3, 4}


def test_sample_raises_on_empty_buffer():
    buf = PrioritizedReplayBuffer(capacity=5)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="empty buffer"):
        buf.sample(batch_size=1, beta=0.4, rng=rng)


def test_sample_raises_when_batch_exceeds_size():
    buf = PrioritizedReplayBuffer(capacity=5)
    buf.add(make_transition(0))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceeds buffer size"):
        buf.sample(batch_size=2, beta=0.4, rng=rng)


def test_sample_returns_correct_shapes():
    buf = PrioritizedReplayBuffer(capacity=10)
    for i in range(10):
        buf.add(make_transition(i))
    rng = np.random.default_rng(0)
    transitions, indices, weights = buf.sample(batch_size=4, beta=0.5, rng=rng)
    assert len(transitions) == 4
    assert indices.shape == (4,)
    assert weights.shape == (4,)
    assert np.all(weights > 0)
    assert weights.max() == pytest.approx(1.0)  # normalized by max


def test_higher_priority_is_sampled_more_often():
    buf = PrioritizedReplayBuffer(capacity=4, alpha=1.0)
    for i in range(4):
        buf.add(make_transition(i))
    # Give transition at index 0 a much higher priority than the rest.
    buf.update_priorities(np.array([0, 1, 2, 3]), np.array([100.0, 0.01, 0.01, 0.01]))

    rng = np.random.default_rng(0)
    counts = np.zeros(4)
    for _ in range(500):
        _, indices, _ = buf.sample(batch_size=1, beta=0.5, rng=rng)
        counts[indices[0]] += 1

    assert counts[0] > counts[1:].sum()  # index 0 dominates sampling


def test_update_priorities_tracks_max_for_new_transitions():
    buf = PrioritizedReplayBuffer(capacity=4)
    buf.add(make_transition(0))
    buf.update_priorities(np.array([0]), np.array([50.0]))
    assert buf._max_priority == pytest.approx(50.0 + buf.eps)

    # A newly added transition should get the (updated) max priority so it
    # is guaranteed to be sampled at least once.
    buf.add(make_transition(1))
    assert buf._priorities[1] == pytest.approx(buf._max_priority)
