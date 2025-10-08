# Network-Bias-Dynamics

This repository investigates how **network structure** influences the **propagation and persistence of local biases** in distributed systems.  
The simulations explore how a single biased node, local noise processes, and coupling topologies (regular, random, and small-world) interact to generate measurable global drifts.

## Motivation

When information, estimates, or control signals are exchanged across a network of nodes, the network topology implicitly defines *who influences whom*.  
Even small structural variations — such as shortcut edges or hubs — can amplify or suppress systematic deviations.  
Here, we study this phenomenon in a controlled setting, using simplified consensus dynamics with additive noise and localized biases.

The goal is to quantify:
- how **topological heterogeneity** (e.g. small-world shortcuts, hub dominance) affects bias propagation,
- how **noise and local averaging** interact to mitigate or amplify these effects,
- and what **robust architectures** minimize global drift.

## Model Summary

Each node \(i\) maintains a scalar state \(x_i(t)\) and updates it by averaging with its neighbors:

\[
x_i(t+1) = (1 - \mu) \left( \frac{1}{|N_i|+1} \left[ x_i(t) + \sum_{j \in N_i} x_j(t) \right] \right)
          + \mu\, b_i + \eta_i(t)
\]

- \(b_i\): local bias (often nonzero for a single node)
- \(\eta_i(t)\): additive noise, independent across nodes (IID)
- \(\mu\): bias weight
- \(N_i\): neighborhood of node \(i\)

We track the **ensemble mean**
\[
\bar{x}(t) = \frac{1}{N} \sum_i x_i(t)
\]
and its **time-averaged shift**, comparing architectures:

| Topology | Description | Mean Degree | Notes |
|-----------|--------------|--------------|-------|
| Regular | Ring lattice | 10 | Uniform connectivity |
| Random | Erdős–Rényi | ≈10 | Homogeneous on average |
| Small-World | Ring + shortcuts | 10–20 | Mix of local and long-range edges |

## Results Overview

- Bias on a **hub node** produces significantly larger global drift than the same bias on a peripheral (low-degree) node.
- **Increasing connectivity** reduces variance but does not eliminate topology-dependent bias amplification.
- Regular networks distribute influence uniformly and yield minimal drift.
- Random and small-world networks show **structural vulnerability**: shortcuts can act as channels for disproportionate influence.

## Getting Started

### Requirements
Python ≥ 3.9 and:
```bash
pip install numpy matplotlib
