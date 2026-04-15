"""Unified character n-gram similarity for Japanese/CJK text.

All similarity calculations in the correx system go through this module.
No other module should implement its own bigram/trigram logic.

Two clean modes:
  - "default": strip whitespace + lowercase (general purpose)
  - "particles": also strip Japanese particles を の が は で に と 、 。
                 (better for semantic comparison of Japanese rules/principles)
"""
from __future__ import annotations

import re

_PARTICLE_RE = re.compile(r"[\s、。をのがはでにと]")
_WHITESPACE_RE = re.compile(r"\s+")


def char_ngrams(text: str, n: int = 2, *, particles: bool = False, normalize_spaces: bool = False) -> set[str]:
    """Extract character n-grams.

    Args:
        text: Input text.
        n: N-gram size (2 for bigram, 3 for trigram).
        particles: If True, strip Japanese particles before n-gram extraction.
        normalize_spaces: If True, collapse whitespace to single space instead
                         of stripping entirely.  Preserves word boundaries in
                         n-grams (e.g. "o w" trigram from "hello world").
    """
    if particles:
        t = _PARTICLE_RE.sub("", text).lower()
    elif normalize_spaces:
        t = _WHITESPACE_RE.sub(" ", text).lower().strip()
    else:
        t = _WHITESPACE_RE.sub("", text).lower()
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int = 2, *, particles: bool = False) -> float:
    """Jaccard similarity of character n-grams.  Returns 0.0–1.0."""
    if not a or not b:
        return 0.0
    sa = char_ngrams(a, n, particles=particles)
    sb = char_ngrams(b, n, particles=particles)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    """Overlap coefficient (intersection / min).  Returns 0.0–1.0.

    More lenient than Jaccard — useful for detecting near-subsets.
    """
    if not a or not b:
        return 0.0
    sa = char_ngrams(a, n)
    sb = char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))
