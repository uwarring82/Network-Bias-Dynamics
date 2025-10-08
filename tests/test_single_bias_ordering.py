from network_bias_dynamics.simulate import single_biased_node_four_traces


def test_time_averaged_mean_ordering_with_strong_bias():
    cfg = dict(
        N=100,
        T=300,
        mu=0.04,
        trials=16,
        bias_level=0.15,
        rng_seed=2024,
        graph_params=dict(
            ring=dict(k=3),
            er=dict(mean_deg=8),
            smallworld_tail=dict(base_k=3, cap_degree=16, extra_attempts_per_node=6),
        ),
    )
    results = single_biased_node_four_traces(**cfg)
    averages = {
        topo: results["topologies"][topo]["time_averages"].mean()
        for topo in ["smallworld_hub", "er", "ring", "smallworld_leaf"]
    }
    tol = 5e-4
    assert averages["smallworld_hub"] >= averages["er"] - tol
    assert averages["er"] >= averages["ring"] - tol
    assert averages["ring"] >= averages["smallworld_leaf"] - tol
