import numpy as np

from network_bias_dynamics.dynamics import step_update_precomp, simulate_mean_traj_precomp
from network_bias_dynamics.graphs import build_segments, ring_neighbors


def _manual_update(x, neighbors, mu, bias, eta_t):
    N = len(neighbors)
    new_x = np.zeros_like(x)
    degrees = np.array([len(n) for n in neighbors], dtype=float)
    for i in range(N):
        if degrees[i] == 0:
            neigh_contrib = x[i]
        else:
            neigh_contrib = 0.0
            for j in neighbors[i]:
                if degrees[j] > 0:
                    neigh_contrib += x[j] / degrees[j]
        new_x[i] = (1 - mu) * x[i] + mu * neigh_contrib + bias[i] + eta_t[i]
    return new_x


def test_step_update_matches_manual_computation():
    N = 6
    neighbors = ring_neighbors(N, 1)
    starts, flat_idx, deg = build_segments(neighbors)
    x = np.linspace(-1.0, 1.0, N)
    mu = 0.2
    bias = np.zeros(N)
    eta_t = np.random.default_rng(0).normal(scale=1e-3, size=N)

    manual = _manual_update(x, neighbors, mu, bias, eta_t)
    precomp = step_update_precomp(x, starts, flat_idx, deg, mu, bias, eta_t)
    assert np.allclose(precomp, manual)


def test_simulate_mean_traj_precomp_shape_and_finiteness():
    N, T = 10, 50
    mu = 0.05
    neighbors = ring_neighbors(N, 2)
    segments = build_segments(neighbors)
    eta = np.zeros((T, N))
    bias = np.zeros(N)
    traj = simulate_mean_traj_precomp(N, *segments, mu, bias, eta)
    assert traj.shape == (T,)
    assert np.all(np.isfinite(traj))
