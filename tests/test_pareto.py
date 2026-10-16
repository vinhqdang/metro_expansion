import numpy as np
import pytest

from src.eval.normalization import Norm, load_norm, save_norm
from src.eval.pareto import eum, hypervolume, igd_plus, nondominated
from src.eval.weights import lattice


def test_nondominated_filters_dominated_point():
    F = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 2.0]])
    np.testing.assert_array_equal(nondominated(F), [True, False, True])


def test_nondominated_keeps_duplicates():
    F = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(nondominated(F), [True, True])


def test_hypervolume_single_point():
    assert hypervolume(np.array([[0.5, 0.5, 0.5]])) == pytest.approx(0.125)


def test_hypervolume_two_points_union():
    F = np.array([[0.5, 0.5, 0.5], [0.8, 0.2, 0.2]])
    # 0.125 + 0.8*0.2*0.2 - overlap 0.5*0.2*0.2 = 0.137
    assert hypervolume(F) == pytest.approx(0.137)


def test_hypervolume_ignores_dominated_and_below_reference():
    base = np.array([[0.5, 0.5, 0.5]])
    assert hypervolume(np.vstack([base, [[0.3, 0.3, 0.3]]])) == pytest.approx(0.125)
    assert hypervolume(np.array([[-1.0, 1.0, 1.0]])) == 0.0


def test_hypervolume_matches_monte_carlo():
    rng = np.random.default_rng(123)
    F = rng.uniform(0.0, 1.0, size=(8, 3))
    samples = rng.uniform(0.0, 1.0, size=(400_000, 3))
    dominated = np.zeros(samples.shape[0], dtype=bool)
    for p in F:
        dominated |= np.all(samples <= p, axis=1)
    mc = dominated.mean()
    assert hypervolume(F) == pytest.approx(mc, abs=0.005)


def test_eum_hand_computed_with_identity_tchebycheff():
    def g(f, w):
        # z = 1, rho = 0: g = max_j w_j (1 - f_j)
        return np.max(np.asarray(w) * (1.0 - np.asarray(f)), axis=-1)

    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W = np.array([[0.5, 0.5, 0.0]])
    assert eum(F, W, g) == pytest.approx(-0.5)
    W2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert eum(F, W2, g) == pytest.approx(0.0)


def test_eum_uses_norm_g():
    norm = Norm(f_ref=np.ones(3), z=1.05, rho=0.05)
    F = np.array([[1.0, 0.0, 0.0]])
    # w=(1,0,0): d=(0.05,0,0) -> g = 0.05 + 0.05*0.05 = 0.0525
    assert eum(F, np.array([[1.0, 0.0, 0.0]]), norm.g) == pytest.approx(-0.0525)


def test_igd_plus_hand_computed():
    F = np.array([[1.0, 1.0, 1.0]])
    assert igd_plus(F, np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])) == pytest.approx(0.0)
    # deficits (1,0,0) and (0,0,2) -> distances 1 and 2 -> mean 1.5
    assert igd_plus(F, np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 3.0]])) == pytest.approx(1.5)


def test_lattice_has_fifteen_vectors_on_simplex():
    W = lattice(H=4, m=3)
    assert W.shape == (15, 3)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert np.all(W >= 0)
    assert len({tuple(w) for w in W}) == 15


def test_norm_normalize_clips_and_roundtrips(tmp_path):
    norm = Norm(f_ref=np.array([0.02, 0.5, 0.1]))
    out = norm.normalize({"D_tot": 0.01, "ggi2": 0.9, "A_new": 0.05})
    np.testing.assert_allclose(out, [0.5, 1.2, 0.5])
    np.testing.assert_allclose(norm.normalize([-0.1, 0.0, 0.1]), [0.0, 0.0, 1.0])
    path = tmp_path / "norm.json"
    save_norm(norm, path)
    back = load_norm(path)
    np.testing.assert_allclose(back.f_ref, norm.f_ref)
    assert back.z == norm.z and back.rho == norm.rho
