"""Noise generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseCfg:
    kind: str = "iid"
    sigma: float = 1e-4
    rho: float = 0.6
    sigma_common: float = 2e-4


def gen_noise_iid(
    cfg: NoiseCfg, N: int, T: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate IID Gaussian noise."""

    if cfg.kind != "iid":
        raise ValueError(f"Unsupported noise kind: {cfg.kind}")
    return rng.normal(loc=0.0, scale=cfg.sigma, size=(T, N))
