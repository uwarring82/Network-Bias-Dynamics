"""Core averaging dynamics."""

from __future__ import annotations

import numpy as np


def step_update_precomp(
    x: np.ndarray,
    starts: np.ndarray,
    flat_idx: np.ndarray,
    deg: np.ndarray,
    mu: float,
    b: np.ndarray,
    eta_t: np.ndarray,
) -> np.ndarray:
    """Perform one update step of the averaging dynamics."""

    N = x.shape[0]
    next_x = np.empty_like(x, dtype=float)
    for i in range(N):
        d = int(deg[i])
        if d > 0:
            idx = flat_idx[starts[i] : starts[i + 1]]
            neigh_mean = x[idx].mean()
        else:
            neigh_mean = x[i]
        next_x[i] = (1.0 - mu) * x[i] + mu * neigh_mean + b[i] + eta_t[i]
    return next_x


def simulate_mean_traj_precomp(
    N: int,
    starts: np.ndarray,
    flat_idx: np.ndarray,
    deg: np.ndarray,
    mu: float,
    b: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Simulate the dynamics and return the mean opinion trajectory."""

    T = eta.shape[0]
    x = np.zeros(N, dtype=float)
    mean_traj = np.empty(T, dtype=float)
    for t in range(T):
        x = step_update_precomp(x, starts, flat_idx, deg, mu, b, eta[t])
        mean_traj[t] = x.mean()
    return mean_traj
