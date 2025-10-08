import numpy as np

from network_bias_dynamics.dynamics import simulate_mean_traj_precomp
from network_bias_dynamics.graphs import (
    build_segments,
    ring_neighbors,
    smallworld_neighbors,
)
from network_bias_dynamics.noise import NoiseCfg, gen_noise_iid
from network_bias_dynamics.simulate import single_biased_node_four_traces


def test_identical_noise_same_mean_trace():
    N, T = 30, 60
    mu = 0.15
    rng = np.random.default_rng(0)
    eta = gen_noise_iid(NoiseCfg(sigma=1e-3), N, T, rng)
    ring_seg = build_segments(ring_neighbors(N, 2))
    sw_seg = build_segments(smallworld_neighbors(N, 2, 0.0, rng))
    traj_ring = simulate_mean_traj_precomp(N, *ring_seg, mu, np.zeros(N), eta)
    traj_sw = simulate_mean_traj_precomp(N, *sw_seg, mu, np.zeros(N), eta)
    assert np.allclose(traj_ring, traj_sw)


def test_single_biased_node_ordering():
    cfg = dict(
        N=80,
        T=200,
        mu=0.05,
        trials=30,
        bias_level=0.12,
        rng_seed=10,
        graph_params=dict(
            ring=dict(k=3),
            er=dict(mean_deg=6),
            smallworld_tail=dict(base_k=3, cap_degree=12, extra_attempts_per_node=5),
        ),
    )
    results = single_biased_node_four_traces(**cfg)
    averages = {
        topo: results["topologies"][topo]["time_averages"].mean()
        for topo in ["ring", "smallworld_hub", "smallworld_leaf"]
    }
    assert averages["smallworld_hub"] >= averages["ring"] - 5e-4
    assert averages["smallworld_leaf"] <= averages["smallworld_hub"] + 5e-4
