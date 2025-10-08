import numpy as np

from network_bias_dynamics.simulate import compare_with_consistent_noise


def test_identical_noise_produces_matching_mean_traces():
    cfg = dict(
        N=40,
        T=80,
        mu=0.1,
        trials=4,
        bias_level=0.0,
        rng_seed=24,
        graph_params=dict(
            ring=dict(k=2),
            er=dict(mean_deg=4),
            smallworld=dict(base_k=2, add_prob=0.05),
        ),
    )
    results = compare_with_consistent_noise(**cfg)
    ring = results["topologies"]["ring"]["trajectories"]
    er = results["topologies"]["er"]["trajectories"]
    sw = results["topologies"]["smallworld"]["trajectories"]

    assert np.allclose(ring, er, atol=1e-10)
    assert np.allclose(ring, sw, atol=1e-10)

    ring_avg = results["topologies"]["ring"]["time_averages"]
    er_avg = results["topologies"]["er"]["time_averages"]
    sw_avg = results["topologies"]["smallworld"]["time_averages"]
    assert np.allclose(ring_avg, er_avg, atol=1e-12)
    assert np.allclose(ring_avg, sw_avg, atol=1e-12)
