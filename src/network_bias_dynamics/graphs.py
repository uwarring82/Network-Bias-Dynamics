"""Graph construction utilities for bias dynamics simulations."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _ensure_array_neighbors(neighbors: Sequence[Iterable[int]]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for neigh in neighbors:
        arr = np.array(sorted(set(neigh)), dtype=np.int64)
        arrays.append(arr)
    return arrays


def ring_neighbors(N: int, k: int) -> List[np.ndarray]:
    """Build a 1-D ring lattice where each node connects to ``2k`` nearest neighbours."""

    if N <= 0:
        raise ValueError("N must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    neighbors: List[set[int]] = [set() for _ in range(N)]
    for offset in range(1, k + 1):
        for i in range(N):
            j = (i + offset) % N
            neighbors[i].add(j)
            neighbors[j].add(i)
    return _ensure_array_neighbors(neighbors)


def er_neighbors(N: int, mean_deg: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Erdős–Rényi graph with expected degree ``mean_deg``."""

    if N <= 0:
        raise ValueError("N must be positive")
    if not 0 <= mean_deg <= N - 1:
        raise ValueError("mean_deg must be within [0, N-1]")
    p = mean_deg / (N - 1) if N > 1 else 0.0
    neighbors: List[set[int]] = [set() for _ in range(N)]
    for i in range(N - 1):
        rand_vals = rng.random(N - i - 1)
        targets = np.where(rand_vals < p)[0] + i + 1
        for j in targets:
            neighbors[i].add(int(j))
            neighbors[j].add(i)
    return _ensure_array_neighbors(neighbors)


def _initial_ring_sets(N: int, base_k: int) -> List[set[int]]:
    neighbors = [set() for _ in range(N)]
    for offset in range(1, base_k + 1):
        for i in range(N):
            j = (i + offset) % N
            neighbors[i].add(j)
            neighbors[j].add(i)
    return neighbors


def smallworld_neighbors(
    N: int, base_k: int, add_prob: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Watts–Strogatz style small-world graph with shortcut edges."""

    if not 0 <= add_prob <= 1:
        raise ValueError("add_prob must be in [0, 1]")
    neighbors = _initial_ring_sets(N, base_k)
    for i in range(N):
        if rng.random() < add_prob:
            existing = neighbors[i]
            candidates = [j for j in range(N) if j != i and j not in existing]
            if not candidates:
                continue
            j = int(rng.choice(candidates))
            neighbors[i].add(j)
            neighbors[j].add(i)
    return _ensure_array_neighbors(neighbors)


def smallworld_tail_neighbors(
    N: int,
    base_k: int,
    cap_degree: int,
    extra_attempts_per_node: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Small-world variant with heavier-tailed degree distribution."""

    if cap_degree <= 0:
        raise ValueError("cap_degree must be positive")
    neighbors = _initial_ring_sets(N, base_k)
    degrees = np.array([len(s) for s in neighbors], dtype=np.int64)
    for i in range(N):
        attempts = 0
        while attempts < extra_attempts_per_node and degrees[i] < cap_degree:
            weights = degrees.astype(float) + 1.0
            weights[i] = 0.0
            mask_cap = degrees < cap_degree
            weights *= mask_cap
            total = weights.sum()
            if total <= 0:
                break
            j = int(rng.choice(N, p=weights / total))
            if j == i or j in neighbors[i]:
                attempts += 1
                continue
            neighbors[i].add(j)
            neighbors[j].add(i)
            degrees[i] += 1
            degrees[j] += 1
            attempts += 1
    return _ensure_array_neighbors(neighbors)


def build_segments(
    neigh_lists: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten neighbour lists for vectorised averaging updates."""

    N = len(neigh_lists)
    starts = np.zeros(N + 1, dtype=np.int64)
    flat_arrays: List[np.ndarray] = []
    for i, neigh in enumerate(neigh_lists):
        arr = np.asarray(neigh, dtype=np.int64)
        flat_arrays.append(arr)
        starts[i + 1] = starts[i] + arr.size
    if flat_arrays:
        flat_idx = np.concatenate(flat_arrays)
    else:
        flat_idx = np.array([], dtype=np.int64)
    degrees = np.diff(starts)
    return starts, flat_idx, degrees
