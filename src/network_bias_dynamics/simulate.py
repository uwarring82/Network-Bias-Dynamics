"""Simulation entry points for experiments."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .analysis import time_averaged_mean_shift
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
    starts, flat_idx, deg = build_segments(neigh)
    return starts, flat_idx, deg.astype(np.int64, copy=False)


def _maybe_store_degrees(target: Dict[str, object], degrees: np.ndarray) -> None:
    if target.get("degree_sequence") is None:
        target["degree_sequence"] = degrees.astype(np.int64)


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
    storage: Dict[str, Dict[str, object]] = {
        topo: {"trajectories": [], "time_averages": [], "degree_sequence": None}
        for topo in topo_names
    }

    ring_params = graph_params["ring"]
    ring_neigh = ring_neighbors(N, int(ring_params["k"]))
    ring_segments = _prepare_segments(ring_neigh)
    _maybe_store_degrees(storage["ring"], ring_segments[2])
    bias_node = 0 if bias_level else None
    shared_bias = _bias_vector(N, bias_level, node=bias_node)

    for _ in range(trials):
        eta = gen_noise_iid(noise_cfg, N, T, rng)

        ring_traj = simulate_mean_traj_precomp(N, *ring_segments, mu, shared_bias, eta)
        storage["ring"]["trajectories"].append(ring_traj)
        storage["ring"]["time_averages"].append(time_averaged_mean_shift(ring_traj))

        trial_seed = int(rng.integers(0, 2**63))
        trial_rng = np.random.default_rng(trial_seed)

        # Erdős–Rényi graph sampled per trial
        er_rng = np.random.default_rng(trial_rng.integers(0, 2**63))
        er_neigh = er_neighbors(N, float(graph_params["er"]["mean_deg"]), er_rng)
        er_segments = _prepare_segments(er_neigh)
        _maybe_store_degrees(storage["er"], er_segments[2])
        er_traj = simulate_mean_traj_precomp(N, *er_segments, mu, shared_bias, eta)
        storage["er"]["trajectories"].append(er_traj)
        storage["er"]["time_averages"].append(time_averaged_mean_shift(er_traj))

        # Small-world graph sampled per trial
        sw_rng = np.random.default_rng(trial_rng.integers(0, 2**63))
        sw_params = graph_params["smallworld"]
        sw_neigh = smallworld_neighbors(
            N,
            int(sw_params["base_k"]),
            float(sw_params["add_prob"]),
            sw_rng,
        )
        sw_segments = _prepare_segments(sw_neigh)
        _maybe_store_degrees(storage["smallworld"], sw_segments[2])
        sw_traj = simulate_mean_traj_precomp(N, *sw_segments, mu, shared_bias, eta)
        storage["smallworld"]["trajectories"].append(sw_traj)
        storage["smallworld"]["time_averages"].append(time_averaged_mean_shift(sw_traj))

    for topo in topo_names:
        storage[topo]["trajectories"] = np.vstack(storage[topo]["trajectories"])
        storage[topo]["time_averages"] = np.asarray(
            storage[topo]["time_averages"], dtype=float
        )
        if storage[topo]["degree_sequence"] is not None:
            storage[topo]["degree_sequence"] = np.asarray(
                storage[topo]["degree_sequence"], dtype=np.int64
            )

    return {
        "time": np.arange(T, dtype=int),
        "topologies": storage,
        "config": {
            "N": N,
            "T": T,
            "mu": mu,
            "trials": trials,
            "bias_level": bias_level,
            "rng_seed": rng_seed,
        },
        "noise_cfg": noise_cfg,
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
    master_rng = np.random.default_rng(rng_seed)
    noise_rng = np.random.default_rng(master_rng.integers(0, 2**63))

    ring_neigh = ring_neighbors(N, int(graph_params["ring"]["k"]))
    ring_segments = _prepare_segments(ring_neigh)

    er_rng = np.random.default_rng(master_rng.integers(0, 2**63))
    er_neigh = er_neighbors(N, float(graph_params["er"]["mean_deg"]), er_rng)
    er_segments = _prepare_segments(er_neigh)

    sw_params = graph_params["smallworld_tail"]
    sw_rng = np.random.default_rng(master_rng.integers(0, 2**63))
    sw_neigh = smallworld_tail_neighbors(
        N,
        int(sw_params["base_k"]),
        int(sw_params["cap_degree"]),
        int(sw_params["extra_attempts_per_node"]),
        sw_rng,
    )
    sw_segments = _prepare_segments(sw_neigh)
    sw_degrees = sw_segments[2]
    hub_node = int(np.argmax(sw_degrees))
    leaf_node = int(np.argmin(sw_degrees))

    storage: Dict[str, Dict[str, object]] = {
        topo: {"trajectories": [], "time_averages": []}
        for topo in ["ring", "er", "smallworld_hub", "smallworld_leaf"]
    }

    bias_ring = _bias_vector(N, bias_level, node=0)
    bias_er = _bias_vector(N, bias_level, node=0)
    bias_hub = _bias_vector(N, bias_level, node=hub_node)
    bias_leaf = _bias_vector(N, bias_level, node=leaf_node)

    for _ in range(trials):
        eta = gen_noise_iid(noise_cfg, N, T, noise_rng)

        ring_traj = simulate_mean_traj_precomp(N, *ring_segments, mu, bias_ring, eta)
        storage["ring"]["trajectories"].append(ring_traj)
        storage["ring"]["time_averages"].append(time_averaged_mean_shift(ring_traj))

        er_traj = simulate_mean_traj_precomp(N, *er_segments, mu, bias_er, eta)
        storage["er"]["trajectories"].append(er_traj)
        storage["er"]["time_averages"].append(time_averaged_mean_shift(er_traj))

        hub_traj = simulate_mean_traj_precomp(N, *sw_segments, mu, bias_hub, eta)
        storage["smallworld_hub"]["trajectories"].append(hub_traj)
        storage["smallworld_hub"]["time_averages"].append(
            time_averaged_mean_shift(hub_traj)
        )

        leaf_traj = simulate_mean_traj_precomp(N, *sw_segments, mu, bias_leaf, eta)
        storage["smallworld_leaf"]["trajectories"].append(leaf_traj)
        storage["smallworld_leaf"]["time_averages"].append(
            time_averaged_mean_shift(leaf_traj)
        )

    for topo in storage:
        storage[topo]["trajectories"] = np.vstack(storage[topo]["trajectories"])
        storage[topo]["time_averages"] = np.asarray(
            storage[topo]["time_averages"], dtype=float
        )

    metadata = {
        "hub_index": hub_node,
        "hub_degree": int(sw_degrees[hub_node]) if sw_degrees.size else 0,
        "leaf_index": leaf_node,
        "leaf_degree": int(sw_degrees[leaf_node]) if sw_degrees.size else 0,
        "degree_sequence": sw_degrees.astype(np.int64),
    }

    return {
        "time": np.arange(T, dtype=int),
        "topologies": storage,
        "config": {
            "N": N,
            "T": T,
            "mu": mu,
            "trials": trials,
            "bias_level": bias_level,
            "rng_seed": rng_seed,
        },
        "noise_cfg": noise_cfg,
        "metadata": metadata,
    }
