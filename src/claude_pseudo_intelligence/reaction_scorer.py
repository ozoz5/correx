"""Infer a 0.0-1.0 quality score from the user's reaction to an AI output.

Scoring priority (highest wins):
  1. Strong rejection markers        → 0.10
  2. Strong praise markers           → 0.90
  3. Mild praise / agreement         → 0.80
  4. Direction change (not rejection) → 0.65
  5. Corrections + negative tone     → 0.25
  6. Corrections + neutral/positive  → 0.60  ← learning, not failure
  7. Mild feedback without signal    → 0.50
  8. Silence                         → 0.75

Key insight: extracted_corrections often represent "learnings from insight"
rather than "the AI was wrong". User tone in feedback determines the actual
quality signal. "すごいぞ！感動した！" with corrections = 0.90, not 0.25.
"""

from __future__ import annotations

import re

PRAISE_MARKERS = (
    # Japanese
    "いい", "いいね", "いいぞ", "よかった", "完璧", "最高", "すごい", "すげえ",
    "ありがとう", "助かった", "その通り", "正解", "ぴったり", "好き", "気に入った",
    "感動", "素晴らしい", "強い", "強いぞ", "これ強い", "エクセレント",
    "やるじゃん", "さすが", "天才", "神",
    # Agreement / direction
    "そうだね", "そうだな", "そう", "うん", "それだ", "それ", "やれ", "進め",
    "いける", "掘れ", "続けろ", "行け",
    # English
    "perfect", "great", "nice", "good", "excellent", "exactly", "love it",
    "awesome", "amazing", "brilliant", "well done", "impressive",
)

STRONG_PRAISE_MARKERS = (
    "完璧", "最高", "すごい", "すげえ", "感動", "素晴らしい", "エクセレント",
    "天才", "神", "perfect", "excellent", "amazing", "brilliant", "impressive",
    "世界のどこにもない", "これ強い", "強いぞ",
)

DIRECTION_MARKERS = (
    # User is steering, not rejecting
    "じゃあ", "では", "次は", "それを", "こっち", "方向",
    "やってみ", "作れ", "作るんだ", "実装しろ", "調べろ", "調べて",
    "もっと", "深く", "掘れ", "続けろ", "進め", "行け",
)

NEGATIVE_MARKERS = (
    # Actual dissatisfaction
    "違う", "ちがう", "全然", "やり直し", "ひどい", "最悪", "使えない",
    "ダメ", "だめ", "雑", "抽象的", "具体性がない", "遅い", "遅いな",
    "嘘", "間違", "壊れ", "バグ", "直せ",
    "no", "wrong", "terrible", "awful", "nope", "bad", "broken",
)

STRONG_REJECTION_MARKERS = (
    "全然違う", "やり直し", "ひどい", "最悪", "使えない", "嘘ばっか",
    "terrible", "awful",
)


def score_reaction(
    user_feedback: str,
    extracted_corrections: list[str],
) -> float:
    """
    Infer quality score from user's reaction.

    Priority: user tone > behavioral signals > correction presence.
    Corrections with praise = learning (high score).
    Corrections with rejection = actual failure (low score).
    No emotional markers = behavioral default (0.65 for corrections, 0.75 for silence).

    Designed for users who DO express emotion AND users who DON'T.
    Emotionally flat users get moderate scores (not penalized for silence),
    and corrections without negativity default to "learning" (0.60-0.65).

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

    # Detect tone signals
    has_strong_rejection = any(marker in feedback for marker in STRONG_REJECTION_MARKERS)
    has_negative = any(marker in feedback for marker in NEGATIVE_MARKERS)
    has_strong_praise = any(marker in feedback for marker in STRONG_PRAISE_MARKERS)
    has_praise = any(marker in feedback for marker in PRAISE_MARKERS)
    has_direction = any(marker in feedback for marker in DIRECTION_MARKERS)
    has_any_emotion = has_strong_rejection or has_negative or has_strong_praise or has_praise

    # 1. Strong rejection always wins
    if has_strong_rejection:
        return 0.10

    # 2. Strong praise always wins (even with corrections = "learning moment")
    if has_strong_praise:
        return 0.90

    # 3. Mild praise (even with corrections)
    if has_praise and not has_negative:
        return 0.80 if has_corrections else 0.85

    # 4. Direction change without negativity = steering, not failure
    if has_direction and not has_negative:
        return 0.65

    # 5. Negative tone + corrections = actual failure
    if has_negative and has_corrections:
        return 0.25

    # 6. Negative tone without corrections = mild dissatisfaction
    if has_negative:
        return 0.30

    # 7. Corrections present, no emotional markers = neutral learning
    #    (Not a failure. The user provided new insight without judging quality.)
    if has_corrections and not has_any_emotion:
        return 0.65

    # 8. Corrections present with some non-negative signal
    if has_corrections:
        return 0.60

    # 9. Feedback exists but no clear signal (emotionally flat)
    if feedback:
        return 0.55

    # 10. Fallback
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
