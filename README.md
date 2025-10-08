# Network Bias Dynamics

Reproducible simulation suite for studying how small biases propagate through consensus-like averaging dynamics on different network topologies. The model follows a noisy DeGroot-style update

\[
\mathbf{x}_{t+1} = (1-\mu)\mathbf{x}_t + \mu A \mathbf{x}_t + \mathbf{b} + \boldsymbol{\eta}_t,
\]

where `A` encodes neighbour averaging, `\mathbf{b}` injects persistent node biases, and `\boldsymbol{\eta}_t` is IID Gaussian noise.

![Preview of generated figures](figures/compare_topologies.png)

## Features

- Pure NumPy implementations of ring, Erdős–Rényi, and small-world graphs.
- Fast simulations using pre-computed neighbour segments.
- Batch experiments that reuse the same noise forcing across topologies for fair comparisons.
- Convenience analysis helpers that output tidy CSV summaries.
- Publication-ready plotting utilities.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Run the baseline experiments (figures saved under `figures/`):

```bash
python scripts/run_compare_topologies.py --config configs/compare_topologies.yaml
python scripts/run_single_biased_node.py --config configs/single_biased_node.yaml
```

Each command produces a PNG with mean ± SEM trajectories and a CSV summary of time-averaged shifts.

## Repository Layout

```
Network-Bias-Dynamics/
├── configs/                     # YAML experiment configurations
├── figures/                     # Generated figures & CSV summaries
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/                     # Entry-point CLI scripts
├── src/network_bias_dynamics/   # Core simulation package
└── tests/                       # Pytest unit tests
```

## Reproducing the four-trace biased-node experiment

The script `scripts/run_single_biased_node.py` creates a heavy-tailed small-world network per trial, locates the highest- and lowest-degree nodes, and injects a persistent bias of `0.15` into either the hub or the leaf. The resulting trajectories illustrate how network position modulates long-term drift.

To adjust the experiment, edit `configs/single_biased_node.yaml` (e.g., change `bias_level`, number of trials, or the degree cap).

## Development

Install optional tooling:

```bash
pip install -e .[dev]
```

Then run the linters and tests:

```bash
ruff check src
black --check src scripts tests
pytest -q
```

Continuous integration runs these checks across Python 3.9–3.12.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
