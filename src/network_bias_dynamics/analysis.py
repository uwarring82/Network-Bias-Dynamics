"""Analysis helpers for simulation outputs."""

from __future__ import annotations

from typing import Mapping, Tuple

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


def summary_dataframe(time_avg: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Construct a tidy summary table from time-averaged shifts."""

    records = []
    for topo, values in time_avg.items():
        arr = np.asarray(values, dtype=float)
        records.append(
            {
                "topology": topo,
                "mean_shift": arr.mean(),
                "sem": sem(arr),
                "trials": arr.size,
            }
        )
    df = (
        pd.DataFrame.from_records(records)
        .sort_values("topology")
        .reset_index(drop=True)
    )
    return df
