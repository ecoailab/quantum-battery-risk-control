from pathlib import Path
from typing import Final


WORKSPACE_ROOT: Final = Path(__file__).parents[2]


def test_public_docs_match_the_neutral_canonical_scope() -> None:
    paths = (
        WORKSPACE_ROOT / "README.md",
        WORKSPACE_ROOT / "paper" / "README.md",
        WORKSPACE_ROOT / "시뮬레이션" / "README.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for required in (
        "neutral",
        "synthetic quasi-static uncertainty",
        "publication_provenance.json",
        "https://github.com/ecoailab/quantum-battery-risk-control",
        "v1.0.1",
    ):
        assert required in text
    for prohibited in (
        "26.6%",
        "0.950",
        "0.808",
        "YOUR_USERNAME",
        "doi.org/PENDING",
        "upon acceptance",
        "github.com/sangkeum/quantum-battery-rl",
    ):
        assert prohibited not in text


def test_package_metadata_contains_no_unverified_public_url() -> None:
    metadata = (WORKSPACE_ROOT / "시뮬레이션" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "[project.urls]" in metadata
    assert "https://github.com/ecoailab/quantum-battery-risk-control" in metadata
    assert 'license = "MIT"' in metadata
    assert 'readme = "README.md"' in metadata


def test_repository_forces_lf_for_hash_linked_artifacts() -> None:
    attributes = (WORKSPACE_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes.splitlines()
