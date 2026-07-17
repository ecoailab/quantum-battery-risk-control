from pathlib import Path
from typing import Final


WORKSPACE_ROOT: Final = Path(__file__).parents[2]
MANUSCRIPT: Final = WORKSPACE_ROOT / "paper" / "main_pra.tex"
BIBLIOGRAPHY: Final = WORKSPACE_ROOT / "paper" / "submission" / "references.bib"


def test_manuscript_follows_neutral_canonical_contract() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    required = (
        "synthetic quasi-static uncertainty",
        "time-only observation",
        "optimizer seed is the independent statistical replicate",
        "Benjamini--Hochberg",
        "make no robustness-superiority claim for PPO",
        "generated/results_macros.tex",
        "generated/method_summary_table.tex",
        "generated/severity_table.tex",
        "generated/cost_table.tex",
        "mean_cvar_comparison.pdf",
        "heldout_distributions.pdf",
        "uncertainty_coverage.pdf",
        "convergence_cost.pdf",
    )
    assert all(item in text for item in required)
    prohibited = (
        "Pontryagin ceiling",
        "IonQ",
        "cross-platform landscape",
        "two-qubit scaling advantage",
        "26.6%",
        "0.950",
        "0.808",
        "91.2%",
    )
    assert not any(item in text for item in prohibited)


def test_generated_macro_names_are_tex_safe_and_citations_are_corrected() -> None:
    macros = (WORKSPACE_ROOT / "paper" / "generated" / "results_macros.tex").read_text(
        encoding="utf-8"
    )
    assert not any(
        character.isdigit()
        for line in macros.splitlines()[1:]
        for character in line.split("}", 1)[0]
    )
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")
    for expected in (
        "Song, Xiaohui",
        "10.1103/y3qx-cs3r",
        "10.1103/6kwv-z6fx",
        "10.1103/vqnk-kzqg",
        "Upper bounds on charging power and tangible advantage in quantum batteries",
        "10.1063/5.0313289",
    ):
        assert expected in bibliography
