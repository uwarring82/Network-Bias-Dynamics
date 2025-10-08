#!/usr/bin/env python3
"""CLI for the single biased node four-trace experiment."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from network_bias_dynamics.analysis import summarize_single_bias
from network_bias_dynamics.plotting import plot_single_biased_node
from network_bias_dynamics.simulate import single_biased_node_four_traces


def _resolved_noise_cfg(raw_cfg, noise_cfg):
    cfg = dict(asdict(noise_cfg))
    if isinstance(raw_cfg, dict):
        cfg.update(raw_cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/single_biased_node.yaml",
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

    results = single_biased_node_four_traces(**cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_single_biased_node(results, save_path=outdir / "single_biased_node.png")

    summary = summarize_single_bias(
        {trace: data["time_averages"] for trace, data in results["topologies"].items()},
        results["config"],
        results.get("metadata", {}),
    )
    summary.to_csv(outdir / "single_biased_node_summary.csv", index=False)

    resolved_cfg = dict(cfg)
    resolved_cfg["noise_cfg"] = _resolved_noise_cfg(cfg.get("noise_cfg", {}), results["noise_cfg"])
    metadata = dict(results.get("metadata", {}))
    if "degree_sequence" in metadata:
        metadata["degree_sequence"] = [int(v) for v in metadata["degree_sequence"]]
    resolved_cfg["metadata"] = metadata
    with open(outdir / "run_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved_cfg, fh, sort_keys=False)


if __name__ == "__main__":
    main()
