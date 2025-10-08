#!/usr/bin/env python3
"""CLI for the single biased node four-trace experiment."""

import argparse

import yaml

from network_bias_dynamics.plotting import plot_single_biased_node
from network_bias_dynamics.simulate import single_biased_node_four_traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/single_biased_node.yaml",
        help="Path to experiment configuration file.",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out = single_biased_node_four_traces(**cfg)
    plot_single_biased_node(out, save_path="figures/single_biased_node.png")


if __name__ == "__main__":
    main()
