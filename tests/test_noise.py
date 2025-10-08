import numpy as np

from network_bias_dynamics.noise import NoiseCfg, gen_noise_iid


def test_gen_noise_iid_statistics():
    rng = np.random.default_rng(42)
    cfg = NoiseCfg(sigma=1e-3)
    eta = gen_noise_iid(cfg, N=25, T=2000, rng=rng)
    assert eta.shape == (2000, 25)
    mean_abs = np.abs(eta.mean())
    assert mean_abs < 5e-5
    corr = np.corrcoef(eta.T)
    off_diag = corr - np.eye(corr.shape[0])
    assert np.max(np.abs(off_diag)) < 0.1
