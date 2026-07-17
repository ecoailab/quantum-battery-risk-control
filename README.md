# Risk-Sensitive Control of a Single-Qubit Quantum Battery

This workspace contains the canonical simulation artifacts and manuscript for a fair open-loop control benchmark under synthetic quasi-static uncertainty.

## Result Branch

The predeclared decision rule selected the **neutral** branch. The tested time-only PPO controller did not satisfy the direction gates for superiority. The paper therefore reports the negative and null comparisons without changing the comparator, metric, or evaluation set.

The retained contribution is narrow:

- one Markovian single-qubit model and ergotropy convention;
- synthetic coherence-time draws fixed within each pulse and varied between episodes;
- disjoint training, validation, and held-out test splits;
- nominal, SAA-mean, CVaR, PPO, bang-bang, sinusoidal, and seeded-random controls;
- optimizer seeds as independent statistical replicates;
- paired held-out evaluation, multiplicity correction, cost accounting, and hash-linked provenance.

No global bound, empirical calibration distribution, measurement-based deployment, experimental-device validation, platform comparison, or multi-qubit scaling result is claimed.

## Reproduce And Verify

From `시뮬레이션/`:

```powershell
uv sync --extra dev
uv run pytest tests -q
uv build
```

From `paper/`, compile the current manuscript:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main_pra.tex
bibtex main_pra
pdflatex -interaction=nonstopmode -halt-on-error main_pra.tex
pdflatex -interaction=nonstopmode -halt-on-error main_pra.tex
```

The main reproducibility surfaces are:

- `시뮬레이션/canonical_manifest.json`: frozen physics, uncertainty, split, budget, and statistical settings.
- `시뮬레이션/results/canonical/`: hash-linked canonical artifacts.
- `시뮬레이션/results/canonical/publication_provenance.json`: publication-value and plot-point mappings.
- `paper/generated/`: deterministic macros and table rows; do not edit manually.
- `paper/figures/canonical/`: four receipt-pinned figures with JSON sidecars.
- `paper/main_pra.tex` and `paper/main_pra.pdf`: current neutral-branch manuscript.

## Repository Layout

```text
paper/                         current manuscript, bibliography, generated values, figures
시뮬레이션/                    Python package, tests, manifest, canonical results
.omo/start-work/               task evidence and immutable execution ledger
.omo/plans/                    publication-revision plan
```

Historical manuscripts and figures remain reference-only. They are not accepted as inputs by the canonical publication pipeline.

## Data And Code Availability

The curated canonical code, tests, results, manuscript sources, and publication figures are publicly available at <https://github.com/ecoailab/quantum-battery-risk-control>. The tagged release and its commit SHA identify the immutable submission snapshot; internal planning records and legacy drafts are intentionally excluded.

## License

The simulation package is distributed under the MIT license. See `LICENSE`.
