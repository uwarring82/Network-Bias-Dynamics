"""Analysis helpers for simulation outputs."""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


def time_averaged_mean_shift(mean_traj: np.ndarray) -> float:
    """Return the time-average of a single mean trajectory."""

    return float(np.mean(mean_traj))


def sem(values: np.ndarray) -> float:
    """Standard error of the mean."""

    n = values.size
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1) with 0 fallback."""

    n = values.size
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def trajectory_mean_and_sem(trials: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and SEM for trajectories stacked as ``(trials, T)``."""

    if trials.ndim != 2:
        raise ValueError("Expected array of shape (trials, T)")
    mean = trials.mean(axis=0)
    if trials.shape[0] <= 1:
        sem_traj = np.zeros_like(mean)
    else:
        sem_traj = trials.std(axis=0, ddof=1) / np.sqrt(trials.shape[0])
    return mean, sem_traj


def degree_histogram(neighbor_lists: Iterable[Iterable[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram counts and bin edges for node degrees."""

    degrees = np.array([len(neigh) for neigh in neighbor_lists], dtype=np.int64)
    if degrees.size == 0:
        return np.array([0]), np.array([0, 1])
    bins = np.arange(degrees.min(), degrees.max() + 2)
    counts, edges = np.histogram(degrees, bins=bins)
    return counts, edges


def summarize_topology_means(
    time_averages: Mapping[str, np.ndarray],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Summarise time-averaged mean shifts for the topology comparison."""

    rows = []
    base = {
        "N": int(config["N"]),
        "T": int(config["T"]),
        "mu": float(config["mu"]),
        "trials": int(config["trials"]),
        "seed": int(config["rng_seed"]),
    }
    for topo, values in time_averages.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "topology": topo,
                "time_avg_mean": float(arr.mean()),
                "std": sample_std(arr),
                "sem": sem(arr),
                **base,
            }
        )
    df = pd.DataFrame(rows, columns=[
        "topology",
        "time_avg_mean",
        "std",
        "sem",
        "N",
        "T",
        "mu",
        "trials",
        "seed",
    ])
    return df.sort_values("topology").reset_index(drop=True)


def summarize_single_bias(
    time_averages: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Summarise results from the single biased node experiment."""

    rows = []
    base = {
        "N": int(config["N"]),
        "T": int(config["T"]),
        "mu": float(config["mu"]),
        "trials": int(config["trials"]),
        "seed": int(config["rng_seed"]),
        "hub_index": int(metadata.get("hub_index", -1)),
        "hub_degree": int(metadata.get("hub_degree", 0)),
        "leaf_index": int(metadata.get("leaf_index", -1)),
        "leaf_degree": int(metadata.get("leaf_degree", 0)),
    }
    for trace, values in time_averages.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "trace": trace,
                "time_avg_mean": float(arr.mean()),
                "std": sample_std(arr),
                "sem": sem(arr),
                **base,
            }
        )
    df = pd.DataFrame(rows, columns=[
        "trace",
        "time_avg_mean",
        "std",
        "sem",
        "N",
        "T",
        "mu",
        "trials",
        "seed",
        "hub_index",
        "hub_degree",
        "leaf_index",
        "leaf_degree",
    ])
    return df.sort_values("trace").reset_index(drop=True)
