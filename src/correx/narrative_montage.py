"""Narrative Montage — generate a user personality narrative from policy data.

Montages opposing policies into a 5-line narrative that describes *who* the
user is, not what rules to follow.  This narrative is auto-written to CLAUDE.md
so the LLM can reason about the user in novel situations where no rule applies.

Architecture:
  - Template-based generation (0 LLM cost per session)
  - LLM regeneration only when policy fingerprint changes
  - Persisted in ~/.correx/narrative.json
  - Written to ~/.claude/CLAUDE.md via inject_guidance.py
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from .schemas import Policy, Tension


# ── Data Model ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class NarrativeState:
    """Persisted state of the generated narrative."""

    narrative_text: str = ""
    policy_fingerprint: str = ""
    generated_at: str = ""
    generation_method: str = ""  # "llm" | "template"
    source_policy_ids: list[str] = field(default_factory=list)


def to_dict(state: NarrativeState) -> dict:
    return asdict(state)


def from_dict(d: dict) -> NarrativeState:
    return NarrativeState(
        narrative_text=d.get("narrative_text", ""),
        policy_fingerprint=d.get("policy_fingerprint", ""),
        generated_at=d.get("generated_at", ""),
        generation_method=d.get("generation_method", ""),
        source_policy_ids=d.get("source_policy_ids", []),
    )


# ── Fingerprint ─────────────────────────────────────────────────────────────


def compute_policy_fingerprint(policies: list[Policy]) -> str:
    """Deterministic hash from active policies.

    Uses evidence_count // 10 so small fluctuations (<10) don't trigger
    regeneration.  A new/removed active policy always changes the fingerprint.
    """
    active = sorted(
        [p for p in policies if p.maturity == "active"],
        key=lambda p: p.id,
    )
    parts = [f"{p.id}:{p.evidence_count // 10}" for p in active]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:12]


def needs_regeneration(current_fp: str, stored: NarrativeState | None) -> bool:
    if stored is None or not stored.narrative_text:
        return True
    return current_fp != stored.policy_fingerprint


# ── Template-Based Narrative Generation ─────────────────────────────────────


def _decision_style(policies: list[Policy], metabolism: float) -> str:
    """Line 1: How this person makes decisions."""
    top = sorted(policies, key=lambda p: p.evidence_count, reverse=True)
    top_title = top[0].title if top else ""

    if metabolism > 0.65:
        if "理解" in top_title:
            return "速く動くが、不確実な局面では必ず確認してから行動する人。"
        return "攻めの判断が速い人。確信があれば即座に動く。"
    elif metabolism < 0.35:
        return "慎重な人。理解が確信に変わるまで動かない。"
    else:
        return "確信度に応じて速度を変える人。確信があれば即動き、なければ確認する。"


def _value_priority(policies: list[Policy]) -> str:
    """Line 2: What this person values most (top 2 policies by evidence)."""
    top2 = sorted(policies, key=lambda p: p.evidence_count, reverse=True)[:2]
    if len(top2) < 2:
        return "まだ十分な修正データがない。"
    return f"最も重視するのは「{top2[0].title}」、次に「{top2[1].title}」。"


def _contradiction_wisdom(tensions: list[Tension]) -> str:
    """Line 3: What this person has learned from opposing principles."""
    active = [t for t in tensions if t.status == "active" and t.boundary]
    if not active:
        return "対立する原則の使い分けはまだ学習途上。"
    t = active[0]  # most representative
    return f"「{t.boundary}」— これがこの人の判断の切り替え基準。"


def _growth_edge(avoidance_keywords: list[str]) -> str:
    """Line 4: What this person learned the hard way."""
    if not avoidance_keywords:
        return "痛い経験からの学びはまだ蓄積されていない。"
    top3 = avoidance_keywords[:3]
    return f"「{'」「'.join(top3)}」に関わる失敗を経験し、そこから学んでいる。"


def _working_style(digestibility: float, reward_keywords: list[str]) -> str:
    """Line 5: How this person prefers to work."""
    style = "具体的で実践的な情報" if digestibility < 0.4 else (
        "抽象的な概念も扱える" if digestibility > 0.6 else "具体と抽象のバランスを取る"
    )
    if reward_keywords:
        top3 = reward_keywords[:3]
        return f"{style}人。「{'」「'.join(top3)}」という言葉に良く反応する。"
    return f"{style}人。"


def build_narrative_template(
    policies: list[Policy],
    tensions: list[Tension],
    *,
    metabolism: float = 0.5,
    digestibility: float = 0.5,
    reward_keywords: list[str] | None = None,
    avoidance_keywords: list[str] | None = None,
) -> str:
    """Build a 5-line personality narrative from template (0 LLM cost).

    Each line captures a different dimension of the user's cognitive profile.
    """
    active = [p for p in policies if p.maturity == "active"]
    if not active:
        return ""

    lines = [
        _decision_style(active, metabolism),
        _value_priority(active),
        _contradiction_wisdom(tensions),
        _growth_edge(avoidance_keywords or []),
        _working_style(digestibility, reward_keywords or []),
    ]
    return "\n".join(lines)


# ── CLAUDE.md Section Markers ───────────────────────────────────────────────

MARKER_BEGIN = "<!-- CORREX:NARRATIVE:BEGIN -->"
MARKER_END = "<!-- CORREX:NARRATIVE:END -->"

SECTION_HEADER = "## お前が仕えている人間"


def format_narrative_section(narrative: str) -> str:
    """Wrap narrative with section markers for CLAUDE.md insertion."""
    return f"""{MARKER_BEGIN}
{SECTION_HEADER}
{narrative}
{MARKER_END}"""


def inject_narrative_into_text(content: str, narrative: str) -> str:
    """Insert or replace the narrative section in a CLAUDE.md file.

    - If markers exist: replace content between them
    - If no markers: prepend at the very top
    """
    section = format_narrative_section(narrative)

    if MARKER_BEGIN in content and MARKER_END in content:
        begin_idx = content.index(MARKER_BEGIN)
        end_idx = content.index(MARKER_END) + len(MARKER_END)
        return content[:begin_idx] + section + content[end_idx:]
    else:
        return section + "\n\n" + content
