"""Render deterministic TeX fragments from canonical paper values."""

from __future__ import annotations

from hashlib import sha256
from re import sub

from .artifact_models import PaperValuesArtifact
from .paper_values_content import METHOD_NAMES


def render_fragments(artifact: PaperValuesArtifact) -> tuple[tuple[str, str], ...]:
    """Render macros and complete table rows without independent numeric inputs."""
    lookup = {value.key: value.formatted for value in artifact.values}
    payload = (artifact.model_dump_json(indent=2) + "\n").encode()
    header = (
        f"% GENERATED; DO NOT EDIT. paper_values_sha256={sha256(payload).hexdigest()}\n"
    )
    macros = header + "".join(
        f"\\newcommand{{\\PV{_macro_name(item.key)}}}{{{item.formatted}}}\n"
        for item in artifact.values
    )
    method_rows = "".join(
        f"{label} & {lookup[f'method.{alias}.mean.estimate']} & {lookup[f'method.{alias}.cvar_0_1.estimate']} & {lookup[f'method.{alias}.minimum.estimate']} \\\\\n"
        for _, alias, label in METHOD_NAMES
    )
    severity_rows = "".join(
        f"{label} & {severity} & {lookup[f'severity.{alias}.s{severity}.mean.estimate']} & {lookup[f'severity.{alias}.s{severity}.cvar_0_1.estimate']} \\\\\n"
        for _, alias, label in METHOD_NAMES
        for severity in (10, 30, 50)
    )
    cost_rows = "".join(
        f"{label} & {lookup[f'cost.{alias}.objective_evaluations']} & {lookup[f'cost.{alias}.environment_steps']} & {lookup[f'cost.{alias}.wall_time_seconds']} \\\\\n"
        for _, alias, label in METHOD_NAMES
    )
    comparison_rows = "".join(
        f"{_tex_key(item.key)} & {item.formatted} \\\\\n"
        for item in artifact.values
        if item.key.startswith("comparison.")
        and item.key.endswith(".mean_difference.estimate")
    )
    return (
        ("results_macros.tex", macros),
        ("method_summary_table.tex", header + method_rows),
        ("severity_table.tex", header + severity_rows),
        ("comparison_table.tex", header + comparison_rows),
        ("cost_table.tex", header + cost_rows),
    )


def _tex_key(key: str) -> str:
    return (
        key.removeprefix("comparison.")
        .removesuffix(".mean_difference.estimate")
        .replace("_", "\\_")
    )


def _macro_name(key: str) -> str:
    compact = sub(r"[^A-Za-z0-9]", "", key.title())
    digit_names = str.maketrans(
        {
            "0": "Zero",
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
        }
    )
    return compact.translate(digit_names)
