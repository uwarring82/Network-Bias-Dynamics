"""Simulation entry points for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from .analysis import summary_dataframe, time_averaged_mean_shift
from .dynamics import simulate_mean_traj_precomp
from .graphs import (
    build_segments,
    er_neighbors,
    ring_neighbors,
    smallworld_neighbors,
    smallworld_tail_neighbors,
)
from .noise import NoiseCfg, gen_noise_iid


def _bias_vector(N: int, bias_level: float, node: int | None = None) -> np.ndarray:
    b = np.zeros(N, dtype=float)
    if node is not None:
        b[node] = bias_level
    return b


def _prepare_segments(neigh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_segments(neigh)


def compare_with_consistent_noise(
    N: int,
    T: int,
    mu: float,
    trials: int,
    bias_level: float,
    graph_params: Mapping[str, Mapping[str, float]],
    rng_seed: int,
    noise_cfg: NoiseCfg | None = None,
) -> Dict[str, object]:
    """Compare topologies under identical stochastic forcing."""

    noise_cfg = noise_cfg or NoiseCfg()
    rng = np.random.default_rng(rng_seed)
    topo_names = ["ring", "er", "smallworld"]
    storage: Dict[str, Dict[str, np.ndarray]] = {
        topo: {"trajectories": [], "time_averages": []} for topo in topo_names
    }
    for _ in range(trials):
        eta = gen_noise_iid(noise_cfg, N, T, rng)
        trial_seed = int(rng.integers(0, 2**63))
        trial_rng = np.random.default_rng(trial_seed)

        # Ring
        ring_neigh = ring_neighbors(N, int(graph_params["ring"]["k"]))
        ring_segments = _prepare_segments(ring_neigh)
        b_ring = _bias_vector(N, bias_level, node=0 if bias_level else None)
        ring_traj = simulate_mean_traj_precomp(N, *ring_segments, mu, b_ring, eta)
        storage["ring"]["trajectories"].append(ring_traj)
        storage["ring"]["time_averages"].append(time_averaged_mean_shift(ring_traj))

        # ER
        er_rng = np.random.default_rng(trial_rng.integers(0, 2**63))
        er_neigh = er_neighbors(N, float(graph_params["er"]["mean_deg"]), er_rng)
        er_segments = _prepare_segments(er_neigh)
        b_er = _bias_vector(N, bias_level, node=0 if bias_level else None)
        er_traj = simulate_mean_traj_precomp(N, *er_segments, mu, b_er, eta)
        storage["er"]["trajectories"].append(er_traj)
        storage["er"]["time_averages"].append(time_averaged_mean_shift(er_traj))

        # Small-world
        sw_rng = np.random.default_rng(trial_rng.integers(0, 2**63))
        sw_params = graph_params["smallworld"]
        sw_neigh = smallworld_neighbors(
            N,
            int(sw_params["base_k"]),
            float(sw_params["add_prob"]),
            sw_rng,
        )
        sw_segments = _prepare_segments(sw_neigh)
        b_sw = _bias_vector(N, bias_level, node=0 if bias_level else None)
        sw_traj = simulate_mean_traj_precomp(N, *sw_segments, mu, b_sw, eta)
        storage["smallworld"]["trajectories"].append(sw_traj)
        storage["smallworld"]["time_averages"].append(time_averaged_mean_shift(sw_traj))

    for topo in topo_names:
        storage[topo]["trajectories"] = np.vstack(storage[topo]["trajectories"])
        storage[topo]["time_averages"] = np.array(
            storage[topo]["time_averages"], dtype=float
        )

    summary = summary_dataframe(
        {topo: data["time_averages"] for topo, data in storage.items()}
    )
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(figures_dir / "compare_topologies_summary.csv", index=False)

    return {
        "time": np.arange(T),
        "topologies": storage,
        "summary": summary,
        "config": {
            "N": N,
            "T": T,
            "mu": mu,
            "trials": trials,
            "bias_level": bias_level,
            "rng_seed": rng_seed,
        },
    }


def single_biased_node_four_traces(
    N: int,
    T: int,
    mu: float,
    trials: int,
    bias_level: float,
    graph_params: Mapping[str, Mapping[str, float]],
    rng_seed: int,
    noise_cfg: NoiseCfg | None = None,
) -> Dict[str, object]:
    """Run the four-trace single-biased-node experiment."""

    noise_cfg = noise_cfg or NoiseCfg()
    rng = np.random.default_rng(rng_seed)
    topo_names = ["ring", "er", "smallworld_hub", "smallworld_leaf"]
    storage: Dict[str, Dict[str, np.ndarray]] = {
        topo: {"trajectories": [], "time_averages": []} for topo in topo_names
    }
    sw_tail_params = graph_params["smallworld_tail"]

    for _ in range(trials):
        eta = gen_noise_iid(noise_cfg, N, T, rng)
        trial_seed = int(rng.integers(0, 2**63))
        trial_rng = np.random.default_rng(trial_seed)

        # Ring topology with bias at node 0
        ring_neigh = ring_neighbors(N, int(graph_params["ring"]["k"]))
        ring_segments = _prepare_segments(ring_neigh)
        b_ring = _bias_vector(N, bias_level, node=0)
        ring_traj = simulate_mean_traj_precomp(N, *ring_segments, mu, b_ring, eta)
        storage["ring"]["trajectories"].append(ring_traj)
        storage["ring"]["time_averages"].append(time_averaged_mean_shift(ring_traj))

        # ER topology with bias at node 0
        er_rng = np.random.default_rng(trial_rng.integers(0, 2**63))
        er_neigh = er_neighbors(N, float(graph_params["er"]["mean_deg"]), er_rng)
        er_segments = _prepare_segments(er_neigh)
        b_er = _bias_vector(N, bias_level, node=0)
        er_traj = simulate_mean_traj_precomp(N, *er_segments, mu, b_er, eta)
        storage["er"]["trajectories"].append(er_traj)
        storage["er"]["time_averages"].append(time_averaged_mean_shift(er_traj))

        # Small-world heavy-tail topology
        sw_rng = np.random.default_rng(trial_rng.integers(0, 2**63))
        sw_neigh = smallworld_tail_neighbors(
            N,
            int(sw_tail_params["base_k"]),
            int(sw_tail_params["cap_degree"]),
            int(sw_tail_params["extra_attempts_per_node"]),
            sw_rng,
        )
        sw_segments = _prepare_segments(sw_neigh)
        degrees = np.array([len(n) for n in sw_neigh])
        hub_node = int(np.argmax(degrees))
        leaf_node = int(np.argmin(degrees))

        b_hub = _bias_vector(N, bias_level, node=hub_node)
        hub_traj = simulate_mean_traj_precomp(N, *sw_segments, mu, b_hub, eta)
        storage["smallworld_hub"]["trajectories"].append(hub_traj)
        storage["smallworld_hub"]["time_averages"].append(
            time_averaged_mean_shift(hub_traj)
        )

        b_leaf = _bias_vector(N, bias_level, node=leaf_node)
        leaf_traj = simulate_mean_traj_precomp(N, *sw_segments, mu, b_leaf, eta)
        storage["smallworld_leaf"]["trajectories"].append(leaf_traj)
        storage["smallworld_leaf"]["time_averages"].append(
            time_averaged_mean_shift(leaf_traj)
        )

    for topo in topo_names:
        storage[topo]["trajectories"] = np.vstack(storage[topo]["trajectories"])
        storage[topo]["time_averages"] = np.array(
            storage[topo]["time_averages"], dtype=float
        )

    summary = summary_dataframe(
        {topo: data["time_averages"] for topo, data in storage.items()}
    )
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(figures_dir / "single_biased_node_summary.csv", index=False)

    return {
        "time": np.arange(T),
        "topologies": storage,
        "summary": summary,
        "config": {
            "N": N,
            "T": T,
            "mu": mu,
            "trials": trials,
            "bias_level": bias_level,
            "rng_seed": rng_seed,
        },
    }
