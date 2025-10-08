from network_bias_dynamics.simulate import (
    compare_with_consistent_noise,
    single_biased_node_four_traces,
)


def test_compare_with_consistent_noise_shapes():
    cfg = dict(
        N=30,
        T=80,
        mu=0.1,
        trials=5,
        bias_level=0.0,
        rng_seed=5,
        graph_params=dict(
            ring=dict(k=2),
            er=dict(mean_deg=4),
            smallworld=dict(base_k=2, add_prob=0.05),
        ),
    )
    results = compare_with_consistent_noise(**cfg)
    assert results["time"].shape == (cfg["T"],)
    for topo in ["ring", "er", "smallworld"]:
        data = results["topologies"][topo]
        assert data["trajectories"].shape == (cfg["trials"], cfg["T"])
        assert data["time_averages"].shape == (cfg["trials"],)
    assert set(results["summary"].columns) == {
        "topology",
        "mean_shift",
        "sem",
        "trials",
    }


def test_single_biased_node_four_traces_outputs():
    cfg = dict(
        N=40,
        T=100,
        mu=0.05,
        trials=6,
        bias_level=0.1,
        rng_seed=7,
        graph_params=dict(
            ring=dict(k=2),
            er=dict(mean_deg=4),
            smallworld_tail=dict(base_k=2, cap_degree=10, extra_attempts_per_node=4),
        ),
    )
    results = single_biased_node_four_traces(**cfg)
    assert results["time"].shape == (cfg["T"],)
    for topo in ["ring", "er", "smallworld_hub", "smallworld_leaf"]:
        data = results["topologies"][topo]
        assert data["trajectories"].shape == (cfg["trials"], cfg["T"])
        assert data["time_averages"].shape == (cfg["trials"],)
