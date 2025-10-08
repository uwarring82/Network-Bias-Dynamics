#!/usr/bin/env python3
"""CLI to compare regular, random, and small-world topologies."""

import argparse
import yaml

from network_bias_dynamics.plotting import plot_compare_topologies
from network_bias_dynamics.simulate import compare_with_consistent_noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/compare_topologies.yaml",
        help="Path to experiment configuration file.",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out = compare_with_consistent_noise(**cfg)
    plot_compare_topologies(out, save_path="figures/compare_topologies.png")


if __name__ == "__main__":
    main()
