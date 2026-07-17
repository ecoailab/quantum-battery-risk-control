from __future__ import annotations

from pathlib import Path
from shutil import copy2, copytree
from typing import Final

import pytest

from quantum_battery_rl.benchmark.provenance_audit import (
    ProvenanceAuditError,
    ProvenanceAuditRequest,
    run_provenance_audit,
)
from quantum_battery_rl.benchmark.provenance_models import (
    PublicationProvenanceArtifact,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
WORKSPACE_ROOT: Final = PROJECT_ROOT.parent
CANONICAL_ROOT: Final = PROJECT_ROOT / "results" / "canonical"
GENERATED_ROOT: Final = WORKSPACE_ROOT / "paper" / "generated"
FIGURE_ROOT: Final = WORKSPACE_ROOT / "paper" / "figures" / "canonical"


def _copy_publication_surface(tmp_path: Path) -> tuple[Path, Path]:
    generated_root = tmp_path / "generated"
    figure_root = tmp_path / "figures"
    _ = copytree(GENERATED_ROOT, generated_root)
    _ = copytree(FIGURE_ROOT, figure_root)
    return generated_root, figure_root


def _request(
    tmp_path: Path,
    generated_root: Path,
    figure_root: Path,
    publication_paths: tuple[Path, ...] = (),
) -> ProvenanceAuditRequest:
    return ProvenanceAuditRequest(
        project_root=PROJECT_ROOT,
        manifest_path=PROJECT_ROOT / "canonical_manifest.json",
        canonical_root=CANONICAL_ROOT,
        generated_root=generated_root,
        figure_root=figure_root,
        output_path=tmp_path / "publication_provenance.json",
        publication_paths=publication_paths,
    )


def test_provenance_audit_maps_every_value_and_plot_point(tmp_path: Path) -> None:
    # Given
    generated_root, figure_root = _copy_publication_surface(tmp_path)
    request = _request(tmp_path, generated_root, figure_root)

    # When
    first = run_provenance_audit(request)
    second = run_provenance_audit(request)

    # Then
    assert not first.reused
    assert second.reused
    assert second.provenance_sha256 == first.provenance_sha256
    artifact = PublicationProvenanceArtifact.model_validate_json(
        request.output_path.read_bytes()
    )
    assert artifact.value_mapping_count == 591
    assert artifact.fragment_count == 5
    assert artifact.figure_count == 4
    assert artifact.plot_point_count == 54_573
    assert artifact.publication_source_sha256 == {}
    assert artifact.unmapped_inputs == ()
    assert artifact.stale_inputs == ()
    assert artifact.legacy_inputs == ()


@pytest.mark.parametrize("legacy_literal", ["0.950", "0.808"])
def test_provenance_audit_rejects_legacy_literal(
    tmp_path: Path,
    legacy_literal: str,
) -> None:
    # Given
    generated_root, figure_root = _copy_publication_surface(tmp_path)
    manuscript = tmp_path / "main.tex"
    _ = manuscript.write_text(
        f"Unsupported legacy result: {legacy_literal}.\n",
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(
        ProvenanceAuditError,
        match=rf"main\.tex:1.*{legacy_literal}",
    ):
        _ = run_provenance_audit(
            _request(tmp_path, generated_root, figure_root, (manuscript,))
        )


def test_provenance_audit_rejects_changed_table_cell(tmp_path: Path) -> None:
    # Given
    generated_root, figure_root = _copy_publication_surface(tmp_path)
    table = generated_root / "cost_table.tex"
    _ = table.write_text(
        table.read_text(encoding="utf-8").replace("26794300", "26794301"),
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(ProvenanceAuditError, match="cost_table.tex"):
        _ = run_provenance_audit(_request(tmp_path, generated_root, figure_root))


def test_provenance_audit_rejects_stale_hash_and_missing_figure(
    tmp_path: Path,
) -> None:
    # Given
    generated_root, figure_root = _copy_publication_surface(tmp_path)
    stale_root = tmp_path / "stale-figures"
    missing_root = tmp_path / "missing-figures"
    _ = copytree(figure_root, stale_root)
    _ = copytree(figure_root, missing_root)
    receipt = stale_root / "figure_receipt.json"
    _ = receipt.write_text(
        receipt.read_text(encoding="utf-8").replace(
            "c48dd3824e4bc0d73486a35f784ca29eb85b0d28deecd58ee335d9b519d88756",
            "f" * 64,
        ),
        encoding="utf-8",
    )
    (missing_root / "mean_cvar_comparison.pdf").unlink()

    # When / Then
    with pytest.raises(ProvenanceAuditError, match="uncertainty_coverage.png"):
        _ = run_provenance_audit(_request(tmp_path, generated_root, stale_root))
    _ = copy2(FIGURE_ROOT / "figure_receipt.json", stale_root / "figure_receipt.json")
    with pytest.raises(ProvenanceAuditError, match="mean_cvar_comparison.pdf"):
        _ = run_provenance_audit(_request(tmp_path, generated_root, missing_root))
