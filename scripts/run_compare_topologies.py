#!/usr/bin/env python3
"""CLI to compare regular, random, and small-world topologies."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from network_bias_dynamics.analysis import summarize_topology_means
from network_bias_dynamics.plotting import plot_compare_topologies
from network_bias_dynamics.simulate import compare_with_consistent_noise


def _resolved_noise_cfg(raw_cfg, noise_cfg):
    cfg = dict(asdict(noise_cfg))
    if isinstance(raw_cfg, dict):
        cfg.update(raw_cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/compare_topologies.yaml",
        help="Path to experiment configuration file.",
    )
    parser.add_argument(
        "--outdir",
        default="figures",
        help="Directory to store figures and CSV outputs.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    results = compare_with_consistent_noise(**cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_compare_topologies(results, save_path=outdir / "compare_topologies.png")

    summary = summarize_topology_means(
        {topo: data["time_averages"] for topo, data in results["topologies"].items()},
        results["config"],
    )
    summary.to_csv(outdir / "compare_topologies_summary.csv", index=False)

    resolved_cfg = dict(cfg)
    resolved_cfg["noise_cfg"] = _resolved_noise_cfg(cfg.get("noise_cfg", {}), results["noise_cfg"])
    with open(outdir / "run_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved_cfg, fh, sort_keys=False)


if __name__ == "__main__":
    main()
