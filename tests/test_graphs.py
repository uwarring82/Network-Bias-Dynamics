import numpy as np

from network_bias_dynamics.graphs import (
    build_segments,
    er_neighbors,
    ring_neighbors,
    smallworld_tail_neighbors,
)


def test_ring_neighbors_symmetry_and_degree():
    N, k = 12, 2
    neighbors = ring_neighbors(N, k)
    for idx, neigh in enumerate(neighbors):
        assert len(neigh) == 2 * k
        for j in neigh:
            assert idx in neighbors[j]


def test_er_neighbors_mean_degree():
    N = 200
    mean_deg = 8
    rng = np.random.default_rng(0)
    neighbors = er_neighbors(N, mean_deg, rng)
    degrees = np.array([len(n) for n in neighbors])
    assert np.all(degrees >= 0)
    assert np.abs(degrees.mean() - mean_deg) < 2.0


def test_smallworld_tail_degree_bounds():
    N = 60
    rng = np.random.default_rng(1)
    base_k = 3
    cap_degree = 12
    neighbors = smallworld_tail_neighbors(N, base_k, cap_degree, 5, rng)
    degrees = np.array([len(n) for n in neighbors])
    assert degrees.max() <= cap_degree
    assert degrees.min() >= 2 * base_k  # ring lattice base degree
    assert degrees.max() > 2 * base_k  # heavy tail produces hubs


def test_build_segments_shapes():
    neighbors = [np.array([1, 2]), np.array([0]), np.array([], dtype=int)]
    starts, flat, deg = build_segments(neighbors)
    assert starts.shape[0] == len(neighbors) + 1
    assert flat.tolist() == [1, 2, 0]
    assert deg.tolist() == [2, 1, 0]
