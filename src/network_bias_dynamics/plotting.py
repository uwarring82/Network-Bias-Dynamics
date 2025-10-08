"""Plotting utilities for simulation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .analysis import trajectory_mean_and_sem


COLORS = {
    "ring": "#1b9e77",
    "er": "#d95f02",
    "smallworld": "#7570b3",
    "smallworld_hub": "#e7298a",
    "smallworld_leaf": "#66a61e",
}


def _plot_with_sem(
    ax, time: np.ndarray, trials: np.ndarray, label: str, color: Optional[str]
) -> float:
    mean, sem = trajectory_mean_and_sem(trials)
    line = ax.plot(time, mean, label=label, color=color)[0]
    ax.fill_between(time, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    ta = float(trials.mean(axis=1).mean())
    ax.axhline(ta, linestyle="--", color=line.get_color(), alpha=0.7)
    return ta


def plot_compare_topologies(results: Dict[str, object], save_path: str | Path) -> None:
    """Plot trajectories comparing the three topologies."""

    time = np.asarray(results["time"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for topo, label in [
        ("ring", "Regular ring"),
        ("er", "Erdős-Rényi"),
        ("smallworld", "Small-world"),
    ]:
        trials = results["topologies"][topo]["trajectories"]
        color = COLORS.get(topo, None)
        _plot_with_sem(ax, time, trials, label, color or None)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Network mean")
    ax.set_title("Bias propagation under identical noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_single_biased_node(results: Dict[str, object], save_path: str | Path) -> None:
    """Plot trajectories for the four-trace biased-node experiment."""

    time = np.asarray(results["time"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for topo, label in [
        ("ring", "Ring (node 0 biased)"),
        ("er", "ER (node 0 biased)"),
        ("smallworld_hub", "Small-world hub biased"),
        ("smallworld_leaf", "Small-world leaf biased"),
    ]:
        trials = results["topologies"][topo]["trajectories"]
        color = COLORS.get(topo, None)
        _plot_with_sem(ax, time, trials, label, color or None)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Network mean")
    ax.set_title("Single biased node trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
