import numpy as np

from network_bias_dynamics.graphs import (
    build_segments,
    er_neighbors,
    ring_neighbors,
    smallworld_tail_neighbors,
)


def test_ring_neighbors_degree_is_two_k():
    N, k = 20, 3
    neighbors = ring_neighbors(N, k)
    assert len(neighbors) == N
    for neigh in neighbors:
        assert len(neigh) == 2 * k


def test_er_neighbors_mean_degree_close():
    N = 400
    mean_deg = 12.0
    rng = np.random.default_rng(123)
    neighbors = er_neighbors(N, mean_deg, rng)
    degrees = np.array([len(n) for n in neighbors], dtype=float)
    observed = degrees.mean()
    assert abs(observed - mean_deg) <= 0.2 * mean_deg


def test_smallworld_tail_respects_degree_cap():
    N = 120
    base_k = 4
    cap_degree = 18
    extra_attempts = 6
    rng = np.random.default_rng(9)
    neighbors = smallworld_tail_neighbors(N, base_k, cap_degree, extra_attempts, rng)
    degrees = np.array([len(n) for n in neighbors])
    assert degrees.max() <= cap_degree
    assert degrees.min() >= 2 * base_k


def test_build_segments_outputs_expected_shapes():
    neighbors = [np.array([1, 2]), np.array([0]), np.array([], dtype=np.int64)]
    starts, flat, deg = build_segments(neighbors)
    assert starts.shape == (len(neighbors) + 1,)
    assert flat.tolist() == [1, 2, 0]
    assert deg.tolist() == [2, 1, 0]
