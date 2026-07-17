# Current Paper Package

## Manuscript

- Source: `main_pra.tex`
- Compiled artifact: `main_pra.pdf`
- Branch: neutral
- Scope: risk-sensitive open-loop control of a single-qubit quantum battery under synthetic quasi-static uncertainty
- Bibliography: `submission/references.bib`

The manuscript imports all scientific result values from `generated/` and uses only the four receipt-pinned figures in `figures/canonical/`.

## Build

Run from `paper/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main_pra.tex
bibtex main_pra
pdflatex -interaction=nonstopmode -halt-on-error main_pra.tex
pdflatex -interaction=nonstopmode -halt-on-error main_pra.tex
```

The verified build produces a 10-page PDF with resolved citations, references, tables, and figures.

## Generated Inputs

- `generated/results_macros.tex`
- `generated/method_summary_table.tex`
- `generated/severity_table.tex`
- `generated/comparison_table.tex`
- `generated/cost_table.tex`
- `figures/canonical/figure_receipt.json`

Generated files must be regenerated from canonical evidence rather than edited manually.

## Submission Route

- Selected journal: Journal of Physics A: Mathematical and Theoretical
- Article type: Paper
- Route: subscription access
- Mandatory author charges: none, as verified from the official journal policy on 2026-07-17
- Decision record: `journal_decision.json`
- Target-specific draft cover letter: `docs/cover_letter_jpa.tex`

Open Systems & Information Dynamics was not selected because its mandatory charges could not be directly verified from the official submission page. Quantum Information Processing remains the fallback.

## Availability

The curated canonical publication archive is public at <https://github.com/ecoailab/quantum-battery-risk-control>. Release `v1.0.2` and its commit SHA identify the submission snapshot. Legacy drafts and internal planning records are excluded from the public package.

## Historical Material

Older QST/PRA drafts, cover letters, and noncanonical figures are submission history only. If historical files conflict with `main_pra.tex`, the canonical manifest, generated fragments, or publication provenance ledger, the canonical artifacts control.
