# Quantum Battery Control Benchmark

Python package for the canonical single-qubit, open-loop control benchmark under synthetic quasi-static uncertainty.

## Environment

- Python 3.9 or newer
- Locked dependencies: `uv.lock`
- Canonical experiment contract: `canonical_manifest.json`

Install and verify from this directory:

```powershell
uv sync --extra dev
uv run pytest tests -q
uv build
```

The reproducible full-suite command is `uv run pytest tests -q`. A bare `pytest -q` is not used because the workspace folder contains square brackets that pytest may interpret as parameter syntax.

## Canonical Artifacts

`results/canonical/` contains the hash-linked draw sets, fairness certificate, fitted-controller references, held-out evaluations, statistical summaries, neutral claim decision, paper values, and publication provenance ledger.

The statistical unit is the optimizer seed. Each seed is evaluated on the same paired held-out draws; draw-level outcomes are not treated as independent replicates.

## Scope

The package supports a simulated Markovian single-qubit benchmark. The uncertainty draws are synthetic, fixed within each pulse, and variable between episodes. The canonical PPO policy receives normalized time only. No empirical device distribution, measurement-based deployment, multi-qubit scaling result, or experimental-device validation is claimed.

## Availability

The curated canonical package is public at <https://github.com/ecoailab/quantum-battery-risk-control>. Release `v1.0.2` and its commit SHA identify the archived submission snapshot.
