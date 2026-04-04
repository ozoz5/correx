from __future__ import annotations

import re

STOPWORDS = {
    "する",
    "した",
    "して",
    "ある",
    "いる",
    "こと",
    "ため",
    "それ",
    "これ",
    "ここ",
    "task",
    "case",
    "project",
    "proposal",
    "analysis",
}

CORRECTION_MARKERS = (
    # --- Japanese ---
    "しろ", "するな", "やめろ", "直せ", "増やせ", "減らせ", "削れ",
    "足りない", "多すぎ", "弱い", "必要", "ダメ", "だめ", "嫌",
    "見切れ", "ズレ", "余白", "整理", "学べ", "ロマン",
    # --- English ---
    "fix", "change", "remove", "add", "update", "replace", "rewrite",
    "don't", "stop", "never", "always", "must", "should",
    "missing", "wrong", "broken", "incorrect",
    "too much", "too little", "too long", "too short",
    "instead", "rather", "better",
    "need", "needs", "require", "required",
    "unnecessary", "redundant", "excessive",
)

EXPLICIT_DIRECTIVE_MARKERS = (
    # --- Japanese ---
    "しろ", "するな", "やめろ", "直せ", "削れ", "避けろ",
    "必ず", "絶対", "常に",
    # --- English ---
    "always", "never", "must", "do not", "don't",
    "stop", "quit", "avoid", "ensure", "make sure",
)

LEADING_FILLERS = (
    "もっと",
    "ちゃんと",
    "まだ",
    "やっぱり",
    "まず",
    "そもそも",
    "とにかく",
)


def normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def extract_keywords(*values: str | None) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.split(r"[\s/・（）()【】「」、。,:：;\-]+", normalize_text(value)):
            token = token.strip()
            if len(token) < 2 or token in STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def normalize_correction_statement(value: str | None) -> str:
    normalized = str(value or "").strip()
    normalized = normalized.replace("！", "").replace("!", "").replace("。", "")
    normalized = re.sub(r"\s+", " ", normalized)
    for filler in LEADING_FILLERS:
        if normalized.startswith(filler):
            normalized = normalized[len(filler) :].strip()
            break
    return normalize_text(normalized)


def is_explicit_directive(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return any(marker in text for marker in EXPLICIT_DIRECTIVE_MARKERS)


def extract_correction_candidates(*values: str | None, limit: int = 5) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for sentence in re.split(r"[\n。！？!?]+", text):
            cleaned = re.sub(r"\s+", " ", sentence).strip(" ・-")
            if len(cleaned) < 4:
                continue
            if not any(marker in cleaned for marker in CORRECTION_MARKERS):
                continue
            normalized = normalize_correction_statement(cleaned)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(cleaned)
            if len(candidates) >= limit:
                return candidates
    if candidates:
        return candidates

    fallback = str(values[0] or "").strip() if values else ""
    if not fallback:
        return []
    normalized = normalize_correction_statement(fallback)
    if not normalized:
        return []
    return [fallback]
