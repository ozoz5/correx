"""Infer a 0.0-1.0 quality score from the user's reaction to an AI output.

Scoring rules:
  silence (no feedback, no corrections)  → 0.75  ← 沈黙が承認
  praise                                 → 0.90
  mild feedback without corrections      → 0.50
  correction present                     → 0.25
  strong rejection                       → 0.10
"""

from __future__ import annotations

PRAISE_MARKERS = (
    "いい",
    "いいね",
    "よかった",
    "完璧",
    "最高",
    "すごい",
    "ありがとう",
    "助かった",
    "その通り",
    "正解",
    "ぴったり",
    "好き",
    "気に入った",
    "perfect",
    "great",
    "nice",
    "good",
    "excellent",
    "exactly",
    "love it",
)

STRONG_REJECTION_MARKERS = (
    "違う",
    "ちがう",
    "全然",
    "やり直し",
    "ひどい",
    "最悪",
    "使えない",
    "no",
    "wrong",
    "terrible",
    "awful",
    "nope",
)


def score_reaction(
    user_feedback: str,
    extracted_corrections: list[str],
) -> float:
    """
    Infer quality score from user's reaction.

    Args:
        user_feedback:        Raw user feedback text (may be empty).
        extracted_corrections: Correction candidates already extracted from the turn.

    Returns:
        float in [0.0, 1.0]
    """
    feedback = (user_feedback or "").strip().lower()
    has_corrections = bool(extracted_corrections)

    # Silence = silent approval
    if not feedback and not has_corrections:
        return 0.75

    # Strong rejection overrides everything
    if any(marker in feedback for marker in STRONG_REJECTION_MARKERS):
        return 0.10

    # Clear praise with no corrections
    if any(marker in feedback for marker in PRAISE_MARKERS) and not has_corrections:
        return 0.90

    # Corrections present → low score
    if has_corrections:
        return 0.25

    # Feedback exists but no strong signal → mild correction assumed
    if feedback:
        return 0.50

    return 0.75


def reaction_label(score: float) -> str:
    """Human-readable label for a reaction score."""
    if score >= 0.85:
        return "praised"
    if score >= 0.65:
        return "accepted"
    if score >= 0.35:
        return "mixed"
    return "corrected"
