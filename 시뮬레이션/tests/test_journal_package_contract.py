from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast


WORKSPACE_ROOT: Final = Path(__file__).parents[2]
PAPER_ROOT: Final = WORKSPACE_ROOT / "paper"


def test_journal_decision_selects_verified_free_subscription_route() -> None:
    decision = cast(
        dict[str, object],
        json.loads((PAPER_ROOT / "journal_decision.json").read_text(encoding="utf-8")),
    )
    assert decision["selected_journal"] == (
        "Journal of Physics A: Mathematical and Theoretical"
    )
    assert decision["selected_route"] == "subscription-access"
    assert decision["mandatory_author_charges"] == 0
    assert decision["ready_for_submission"] is True
    assert decision["submission_blockers"] == []


def test_jpa_cover_letter_is_neutral_target_specific_and_blocked() -> None:
    text = (PAPER_ROOT / "docs" / "cover_letter_jpa.tex").read_text(encoding="utf-8")
    for required in (
        "Journal of Physics A: Mathematical and Theoretical",
        "neutral result branch",
        "did not establish superiority",
        "subscription-access route",
        "https://github.com/ecoailab/quantum-battery-risk-control",
    ):
        assert required in text
    for prohibited in ("PPO advantage", "26.6%", "0.950", "IonQ", "QPU"):
        assert prohibited not in text
