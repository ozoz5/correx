"""GhostEngine — Autonomous learning from rejected alternatives.

When the AI proposes something and the user rejects it, the rejected
option isn't discarded. It persists as a "ghost" — a counterfactual
memory that carries the AI's predicted outcome.

As actual outcomes are observed, prediction errors accumulate.
Ghosts with similar interference themes cluster into trajectories.
When a trajectory's cumulative prediction error crosses the firing
threshold, it fires. Firing triggers sublimation: extracting a
higher-order principle from the pattern of rejections — autonomously,
without requiring any further human correction.

This is the second layer of the Engram engine. The surface layer
(save_conversation_turn → rules → meanings → principles) learns
from explicit corrections. The ghost layer learns from the gap
between what was proposed and what was actually wanted.

Theoretical foundations:
  - HOPE (Cornell ICML 2025): counterfactual hindsight exploration
  - Active Inference (Friston): suppressed predictions generate
    ongoing prediction error = informational value
  - Quantum Cognition (Pothos & Busemeyer): interference term
    2αβcos(θ) measures hesitation strength at decision points
  - CPL (Contrastive Preference Learning): gap between chosen and
    rejected carries the corrective signal
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone

from .schemas import Ghost, GhostTrajectory


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCOLDED_PATTERNS = re.compile(
    r"違う|ちがう|だめ|ダメ|違います|間違い|おかしい|なんで|なぜ|ひどい"
    r"|wrong|no|incorrect|bad|terrible|why|awful|stupid|ugh|wtf",
    re.IGNORECASE,
)

_CORRECTED_PATTERNS = re.compile(
    r"修正|直して|やり直し|変えて|そうじゃなくて|そっちじゃなく"
    r"|fix|change|redo|not like that|instead|rather|edit|revise",
    re.IGNORECASE,
)

# Negation signal in Japanese (correction without explicit scolding)
_NEGATION_PATTERNS = re.compile(
    r"するな|やめろ|避けろ|しないで|しないこと|使わないで|入れないで",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Origin classification
# ---------------------------------------------------------------------------

def detect_origin(user_feedback: str) -> str:
    """Classify the correction origin from user feedback text.

    Returns "scolded" | "corrected" | "rejected".

    "scolded" has the highest signal quality because the AI has no malice —
    the gap between AI intent and user frustration is the cleanest signal
    of authentic user preference that diverges from the AI's model.
    """
    if not user_feedback:
        return "rejected"

    # Check for emotional frustration first (highest signal)
    if _SCOLDED_PATTERNS.search(user_feedback):
        return "scolded"

    # Then explicit correction
    if _CORRECTED_PATTERNS.search(user_feedback) or _NEGATION_PATTERNS.search(user_feedback):
        return "corrected"

    return "rejected"


def origin_weight(origin: str) -> float:
    """Weight multiplier for prediction error based on origin quality.

    scolded: 2.0 — strongest signal, intent/expectation gap is clear
    corrected: 1.5 — clear direction given
    rejected: 1.0 — base signal, direction less explicit
    """
    return {"scolded": 2.0, "corrected": 1.5, "rejected": 1.0}.get(origin, 1.0)


# ---------------------------------------------------------------------------
# Prediction error — text-based divergence (no embeddings required)
# ---------------------------------------------------------------------------

from .text_similarity import char_ngrams as _ts_char_ngrams  # noqa: E402


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    # Ghost engine preserves word boundaries (original behavior)
    return _ts_char_ngrams(text, n, normalize_spaces=True)


def _ngram_similarity(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    sa = _char_ngrams(a, n)
    sb = _char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _length_divergence(predicted: str, actual: str) -> float:
    """Length ratio divergence — proxy for density mismatch.

    If the AI predicted a short response but the user gave a long correction,
    or vice versa, this captures that structural mismatch.
    """
    lp = max(len(predicted.strip()), 1)
    la = max(len(actual.strip()), 1)
    ratio = max(lp, la) / min(lp, la)
    return min(1.0, math.log(ratio) / math.log(10))


def _sentiment_divergence(predicted: str, actual: str) -> float:
    """Polarity divergence — positive predicted vs negative actual feedback.

    Uses simple keyword-based polarity scoring (no ML, no API).
    """
    positive = re.compile(
        r"いい|よい|良い|完璧|素晴らしい|最高|good|great|perfect|yes|exactly|right",
        re.IGNORECASE,
    )
    negative = re.compile(
        r"違う|ダメ|おかしい|悪い|最悪|no|wrong|bad|terrible|incorrect|not",
        re.IGNORECASE,
    )

    def polarity(text: str) -> float:
        pos = len(positive.findall(text))
        neg = len(negative.findall(text))
        total = pos + neg
        if total == 0:
            return 0.5  # neutral
        return pos / total

    pp = polarity(predicted)
    pa = polarity(actual)
    return abs(pp - pa)


def compute_prediction_error(predicted_outcome: str, actual_outcome: str) -> float:
    """Compute prediction error between what was predicted and what happened.

    This is a text-based proxy for KL divergence — no embeddings required.

    Combines three signals:
    1. Semantic divergence: n-gram overlap (inverted — high overlap = low error)
    2. Length divergence: structural mismatch
    3. Sentiment divergence: polarity gap

    Returns float in [0, ∞) — higher = more surprising = higher ghost value.
    The practical range is ~0.0 to ~2.5.
    """
    if not predicted_outcome or not actual_outcome:
        # No prediction stored → assume moderate error
        return 0.5

    semantic_sim = _ngram_similarity(predicted_outcome, actual_outcome)
    semantic_error = 1.0 - semantic_sim  # high sim = low error

    length_err = _length_divergence(predicted_outcome, actual_outcome)
    sentiment_err = _sentiment_divergence(predicted_outcome, actual_outcome)

    # Weighted combination
    return semantic_error * 0.6 + length_err * 0.25 + sentiment_err * 0.15


def compute_interference(
    rejected_output: str,
    accepted_output: str = "",
) -> float:
    """Compute hesitation strength — quantum interference proxy.

    High interference = the AI was genuinely torn between alternatives.
    Low interference = the AI was confident but wrong anyway.

    For the ghost engine, interference determines:
    - How quickly a trajectory reaches firing threshold
    - The weight of this ghost's contribution to sublimated principles

    Practical heuristic (without access to token logprobs):
    - Semantic distance between rejected and accepted outputs
    - Length of rejected output (longer = more committed = less hesitation)
    - Presence of hedging language in rejected output

    Returns float in [0, 1].
    """
    if not rejected_output:
        return 0.3  # default moderate interference

    # Semantic distance between alternatives (if accepted output is known)
    if accepted_output:
        sim = _ngram_similarity(rejected_output, accepted_output)
        # High similarity = small difference = high hesitation territory
        # Very different = the AI was clearly making a choice = lower interference
        # U-shaped: moderate similarity = highest hesitation
        interference_from_sim = 1.0 - abs(2 * sim - 1.0)
    else:
        interference_from_sim = 0.5

    # Length proxy: short rejected outputs suggest uncertain commitment
    length_factor = min(1.0, len(rejected_output) / 200)
    length_interference = 1.0 - length_factor * 0.5

    # Hedging language in rejected output
    hedge_patterns = re.compile(
        r"かもしれ|かも|思われ|考えられ|可能性|might|maybe|perhaps|could|possibly",
        re.IGNORECASE,
    )
    has_hedging = bool(hedge_patterns.search(rejected_output))
    hedge_boost = 0.2 if has_hedging else 0.0

    return min(1.0, interference_from_sim * 0.6 + length_interference * 0.25 + hedge_boost)


# ---------------------------------------------------------------------------
# Trajectory clustering
# ---------------------------------------------------------------------------

def _trajectory_similarity(ghost: Ghost, trajectory: GhostTrajectory) -> float:
    """Similarity between a ghost and a trajectory (for clustering)."""
    # Theme similarity
    theme_sim = _ngram_similarity(ghost.rejected_output, trajectory.theme)

    # Scope match bonus
    scope_bonus = 0.15 if ghost.task_scope and ghost.task_scope in trajectory.scopes else 0.0

    # Interference similarity (ghosts with similar hesitation patterns cluster)
    # We approximate trajectory interference by using the theme length as proxy
    interference_sim = 0.0  # would require stored mean interference per trajectory

    return theme_sim * 0.75 + scope_bonus + interference_sim


def assign_ghost_to_trajectory(
    ghost: Ghost,
    trajectories: list[GhostTrajectory],
    similarity_threshold: float = 0.20,
) -> tuple[GhostTrajectory | None, bool]:
    """Find the best matching open trajectory for this ghost.

    Returns (trajectory, is_new) where:
    - trajectory = the matched or newly created trajectory
    - is_new = True if a new trajectory was created

    New trajectories are created when no existing trajectory matches
    above the similarity threshold.
    """
    best_match: GhostTrajectory | None = None
    best_score: float = 0.0

    for t in trajectories:
        if t.status != "open":
            continue
        score = _trajectory_similarity(ghost, t)
        if score > best_score:
            best_score = score
            best_match = t

    if best_score >= similarity_threshold and best_match is not None:
        return best_match, False

    # Create new trajectory
    now = _now_str()
    trajectory_id = _make_id(f"traj_{ghost.id}_{now}")
    new_trajectory = GhostTrajectory(
        id=trajectory_id,
        created_at=now,
        updated_at=now,
        theme=ghost.rejected_output[:200],  # seed theme from first ghost
        ghost_ids=[],
        cumulative_pe=0.0,
        firing_threshold=1.0,
        fired=False,
        fired_at="",
        sublimated_principle="",
        source_ghost_count=0,
        scopes=[ghost.task_scope] if ghost.task_scope else [],
        origin_mix={},
        status="open",
    )
    return new_trajectory, True


def add_ghost_to_trajectory(ghost: Ghost, trajectory: GhostTrajectory) -> GhostTrajectory:
    """Update a trajectory with a new ghost, applying prediction error and origin weight."""
    weighted_pe = ghost.prediction_error * origin_weight(ghost.origin)

    trajectory.ghost_ids.append(ghost.id)
    trajectory.cumulative_pe += weighted_pe
    trajectory.source_ghost_count = len(trajectory.ghost_ids)
    trajectory.updated_at = _now_str()

    # Update scopes
    if ghost.task_scope and ghost.task_scope not in trajectory.scopes:
        trajectory.scopes.append(ghost.task_scope)

    # Update origin mix
    trajectory.origin_mix[ghost.origin] = trajectory.origin_mix.get(ghost.origin, 0) + 1

    # Refine theme: blend with new ghost content if different scope
    if ghost.task_scope and ghost.task_scope not in (trajectory.scopes[:1] or [""]):
        # Cross-scope ghost — update theme to reflect broader pattern
        if len(trajectory.theme) > 0 and len(ghost.rejected_output) > 0:
            old_theme = trajectory.theme
            new_content = ghost.rejected_output[:150]
            # Simple blending: take common n-gram anchor from both
            trajectory.theme = _blend_themes(old_theme, new_content)

    return trajectory


def _blend_themes(theme_a: str, theme_b: str) -> str:
    """Blend two theme strings by extracting common keywords."""
    words_a = set(re.findall(r"[ぁ-んァ-ンー一-龯a-zA-Z]{2,}", theme_a))
    words_b = set(re.findall(r"[ぁ-んァ-ンー一-龯a-zA-Z]{2,}", theme_b))
    common = words_a & words_b
    if common:
        return " / ".join(sorted(common)[:5])
    # No common words — keep original theme
    return theme_a


# ---------------------------------------------------------------------------
# Firing detection
# ---------------------------------------------------------------------------

def calculate_trajectory_confidence(trajectory: GhostTrajectory) -> float:
    """Calculate confidence that trajectory represents a genuine pattern (not noise).

    Inspired by AgentHER's confidence gating (arXiv:2603.21357).

    Three factors:
    - count_factor:  more ghosts = more confidence (saturates at 5)
    - scope_factor:  cross-scope ghosts = more generalizable (saturates at 3)
    - origin_factor: mixed origins (rejected/scolded/corrected) = robust signal

    Returns:
        Confidence score in [0.0, 1.0].
    """
    count_factor = min(1.0, trajectory.source_ghost_count / 5.0)
    scope_factor = min(1.0, len(trajectory.scopes) / 3.0)
    origin_factor = min(1.0, len(trajectory.origin_mix) / 3.0)
    confidence = 0.5 * count_factor + 0.3 * scope_factor + 0.2 * origin_factor
    return round(confidence, 3)


def should_fire(
    trajectory: GhostTrajectory,
    metabolism_rate: float = 0.5,
    min_confidence: float = 0.3,
) -> bool:
    """Determine if a trajectory should fire based on accumulated prediction error.

    Firing requires:
    1. Cumulative PE >= firing threshold
    2. At least 2 ghosts (single observation is not a pattern)
    3. Trajectory is still open
    4. Confidence score >= min_confidence (noise rejection)

    The firing threshold adapts to metabolism_rate:
    - High metabolism (fast learner) → lower threshold → fires sooner
    - Low metabolism (cautious) → higher threshold → needs more evidence

    The confidence gate rejects noise patterns that accumulate high PE
    quickly but lack scope/origin diversity (e.g., single-session artifacts).
    """
    if trajectory.status != "open":
        return False
    if trajectory.source_ghost_count < 2:
        return False

    # Adaptive threshold: base 1.0, scaled by metabolism
    # metabolism_rate=0.0 → threshold=1.5, 0.5 → 1.0, 1.0 → 0.5
    adaptive_threshold = 1.5 - metabolism_rate
    if trajectory.cumulative_pe < adaptive_threshold:
        return False

    # Confidence gating: reject low-confidence patterns (noise/single-session artifacts)
    confidence = calculate_trajectory_confidence(trajectory)
    return confidence >= min_confidence


# ---------------------------------------------------------------------------
# Sublimation — extracting principles from fired trajectories
# ---------------------------------------------------------------------------

def sublimate(
    trajectory: GhostTrajectory,
    all_ghosts: dict[str, Ghost],
) -> str:
    """Extract an autonomous principle from a fired trajectory.

    Sublimation converts accumulated ghost prediction errors into a
    higher-order behavioral principle — without requiring human input.

    Uses local LLM (ollama) when available for high-quality extraction.
    Falls back to template-based generation if LLM is unavailable.
    """
    ghost_ids = trajectory.ghost_ids
    ghosts = [all_ghosts[gid] for gid in ghost_ids if gid in all_ghosts]

    if not ghosts:
        return ""

    # Try LLM-based sublimation first
    llm_result = _sublimate_with_llm(trajectory, ghosts)
    if llm_result:
        return llm_result

    # Fallback: template-based
    return _sublimate_template(trajectory, ghosts)


def _sublimate_with_llm(
    trajectory: GhostTrajectory,
    ghosts: list[Ghost],
) -> str:
    """Use local LLM to extract a principle from the rejection pattern."""
    import json as _json
    import urllib.request
    import urllib.error

    # Build context for LLM
    entries = []
    for g in ghosts[:5]:  # cap at 5 to fit context
        origin_label = {"scolded": "叱責", "corrected": "修正指示", "rejected": "却下"}.get(g.origin, g.origin)
        entries.append(
            f"[{origin_label}] AI出力: {g.rejected_output[:200]}\n"
            f"  ユーザー反応: {g.actual_outcome[:200]}"
        )

    scopes = trajectory.scopes
    scope_str = "・".join(scopes[:3]) if scopes else "不明"

    prompt = f"""あなたはAIアシスタントの行動分析官です。

以下は、AIアシスタントが出した提案がユーザーに繰り返し却下・修正された記録です。

スコープ: {scope_str}
却下回数: {len(ghosts)}回

--- 却下記録 ---
{"---".join(entries)}
--- 記録ここまで ---

上記の繰り返される却下パターンから、ユーザーが暗黙に求めている「行動ルール」を1つだけ、20〜50文字の日本語で書いてください。
「〜するな」「〜してから〜せよ」のように、具体的で次回から守れる指示にしてください。

行動ルール:"""

    try:
        # Check which models are available, prefer 7b
        model = "qwen2.5:7b"
        try:
            tags_req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(tags_req, timeout=3) as tags_resp:  # nosec B310
                tags = _json.loads(tags_resp.read())
                available = [m["name"] for m in tags.get("models", [])]
                if not available:
                    return ""  # no models at all — fall back to template
                if model not in available:
                    model = available[0]
        except (urllib.error.URLError, OSError, KeyError, _json.JSONDecodeError):
            pass  # assume default model, will fail gracefully below

        body = _json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 80},
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # nosec B310
            result = _json.loads(resp.read())
        raw = result.get("response", "").strip()
        # Clean up: take first line, strip quotes
        principle = raw.split("\n")[0].strip().strip("「」\"'")
        if principle and 3 < len(principle) < 150:
            return principle
    except (urllib.error.URLError, OSError, KeyError, _json.JSONDecodeError):
        pass

    return ""  # fall back to template


def _sublimate_template(
    trajectory: GhostTrajectory,
    ghosts: list[Ghost],
) -> str:
    """Template-based fallback when LLM is unavailable."""
    rejected_texts = [g.rejected_output for g in ghosts if g.rejected_output]
    actual_texts = [g.actual_outcome for g in ghosts if g.actual_outcome]

    rejected_keywords = _extract_common_keywords(rejected_texts)
    wanted_keywords = _extract_common_keywords(actual_texts)

    origin_mix = trajectory.origin_mix
    dominant_origin = max(origin_mix, key=lambda k: origin_mix[k]) if origin_mix else "rejected"

    scopes = trajectory.scopes
    scope_str = "・".join(scopes[:3]) if scopes else "全スコープ"

    return _generate_principle_text(
        trajectory_theme=trajectory.theme,
        rejected_keywords=rejected_keywords,
        wanted_keywords=wanted_keywords,
        dominant_origin=dominant_origin,
        scope_str=scope_str,
        ghost_count=len(ghosts),
        cumulative_pe=trajectory.cumulative_pe,
    )


def _extract_common_keywords(texts: list[str]) -> list[str]:
    """Extract keywords that appear in multiple texts (frequency > 1)."""
    if not texts:
        return []

    word_counter: dict[str, int] = {}
    pattern = re.compile(r"[ぁ-んァ-ンー一-龯]{2,}|[a-zA-Z]{3,}")

    for text in texts:
        words = set(pattern.findall(text))
        for w in words:
            word_counter[w] = word_counter.get(w, 0) + 1

    # Return words appearing in ≥2 texts, sorted by frequency
    return sorted(
        [w for w, count in word_counter.items() if count >= 2],
        key=lambda w: -word_counter[w],
    )[:8]


def _generate_principle_text(
    trajectory_theme: str,
    rejected_keywords: list[str],
    wanted_keywords: list[str],
    dominant_origin: str,
    scope_str: str,
    ghost_count: int,
    cumulative_pe: float,
) -> str:
    """Generate a natural language principle from the ghost trajectory data."""
    confidence_label = (
        "強い確信" if cumulative_pe > 2.0
        else "中程度の確信" if cumulative_pe > 1.0
        else "初期シグナル"
    )

    origin_label = {
        "scolded": "強い修正シグナル（叱責）",
        "corrected": "明示的な修正",
        "rejected": "選択されなかった代替案",
    }.get(dominant_origin, "修正シグナル")

    if rejected_keywords:
        rejected_kw_str = "・".join(rejected_keywords[:4])
        if wanted_keywords:
            wanted_kw_str = "・".join(wanted_keywords[:4])
            principle = (
                f"[{scope_str}] において「{rejected_kw_str}」のある出力は繰り返し修正されてきた。"
                f"代わりに「{wanted_kw_str}」が求められている傾向がある。"
                f"（{ghost_count}件のゴーストから自律抽出、{origin_label}、{confidence_label}）"
            )
        else:
            principle = (
                f"[{scope_str}] において「{rejected_kw_str}」を含む提案は"
                f"繰り返し却下されている。このパターンを回避せよ。"
                f"（{ghost_count}件のゴーストから自律抽出、{origin_label}、{confidence_label}）"
            )
    else:
        # No clear keywords — use theme-based principle
        theme_short = trajectory_theme[:80].strip()
        principle = (
            f"[{scope_str}] において類似した提案が{ghost_count}回却下された。"
            f"テーマ: 「{theme_short}...」"
            f"このパターンについて慎重に再検討せよ。"
            f"（自律抽出、{origin_label}、{confidence_label}）"
        )

    return principle


# ---------------------------------------------------------------------------
# Ghost creation
# ---------------------------------------------------------------------------

def create_ghost(
    *,
    rejected_output: str,
    task_scope: str = "",
    tags: list[str] | None = None,
    user_feedback: str = "",
    accepted_output: str = "",
    source_turn_id: str = "",
) -> Ghost:
    """Create a new Ghost from a rejected AI proposal.

    The actual_outcome and prediction_error are filled immediately using
    the user_feedback as the observed outcome. The predicted_outcome is
    inferred from the accepted_output or left as a neutral prediction.
    """
    now = _now_str()
    ghost_id = _make_id(f"ghost_{now}_{rejected_output[:30]}")

    origin = detect_origin(user_feedback)
    interference = compute_interference(rejected_output, accepted_output)

    # Predicted outcome: what the AI "expected" when generating this proposal
    # In practice, without logprob access, we use the rejected output itself
    # as a proxy for what the AI thought would be accepted
    predicted_outcome = f"承認・採用される（{rejected_output[:100]}）"

    # Actual outcome: what actually happened = user's rejection feedback
    actual_outcome = user_feedback

    # Prediction error: how wrong the AI's prediction was
    prediction_error = compute_prediction_error(predicted_outcome, actual_outcome)

    return Ghost(
        id=ghost_id,
        created_at=now,
        rejected_output=rejected_output,
        predicted_outcome=predicted_outcome,
        interference=interference,
        actual_outcome=actual_outcome,
        prediction_error=prediction_error,  # raw; origin_weight applied in trajectory
        origin=origin,
        task_scope=task_scope,
        tags=list(tags or [])[:10],
        source_turn_id=source_turn_id,
        trajectory_id="",
    )


# ---------------------------------------------------------------------------
# Full ghost processing pipeline
# ---------------------------------------------------------------------------

def _sanitize_principle(text: str) -> str:
    """Remove markdown noise, tables, and excessive formatting from principles."""
    if not text:
        return ""
    # Remove markdown tables
    text = re.sub(r"\|[^\n]*\|", "", text)
    # Remove markdown separators
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove citation-style markers
    text = re.sub(r"\([\d]+件の.*?\)", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Take first sentence only if result is too long
    if len(text) > 80:
        first_sentence = re.split(r"[。\n]", text)[0]
        if first_sentence and len(first_sentence) > 5:
            text = first_sentence
    return text.strip()


# Task-specific terms that indicate a principle is not generalizable
_TASK_SPECIFIC_PATTERNS = re.compile(
    r"ロゴ|ヘッダー|フッター|サムネイル|アップロード|ナレーション|BGM|LP|"
    r"CSV|Excel|PowerPoint|PS1|VM|Parallels|HuggingFace|Streamlit|"
    r"動画|カット\d|スライド|絵コンテ|フォルダ|シート|"
    r"オッズ|発走|成績|マイニング|検診|キット|グルーヴ|合いの手|"
    r"API[キコ]|デプロイ|GitHub|wanは",
    re.IGNORECASE,
)

# Broken/meta principles that are clearly not behavioral rules
_BROKEN_PRINCIPLE_PATTERNS = re.compile(
    r"固有原則|汎用化します|を汎用化すると|解説[:：]|この原則は",
    re.IGNORECASE,
)


def is_principle_generalizable(principle: str) -> bool:
    """Check if a principle is generalizable (not task-specific).

    Returns True if the principle is likely useful across contexts.
    Returns False if it references specific tools, UI elements, or project terms.

    Strategy: reject known-bad patterns rather than requiring known-good patterns.
    Japanese verb conjugation is too complex for positive matching.
    """
    if not principle or len(principle) < 8:
        return False

    # Contains task-specific terms → not generalizable
    if _TASK_SPECIFIC_PATTERNS.search(principle):
        return False

    # Broken/meta text that isn't a real principle
    if _BROKEN_PRINCIPLE_PATTERNS.search(principle):
        return False

    return True


def _is_duplicate_principle(
    new_principle: str,
    existing_principles: list[str],
    threshold: float = 0.35,
) -> bool:
    """Check if new principle is too similar to any existing one."""
    if not new_principle or not existing_principles:
        return False
    new_ngrams = _principle_ngrams(new_principle)
    if not new_ngrams:
        return False
    for existing in existing_principles:
        existing_ngrams = _principle_ngrams(existing)
        if not existing_ngrams:
            continue
        intersection = new_ngrams & existing_ngrams
        union = new_ngrams | existing_ngrams
        similarity = len(intersection) / len(union) if union else 0.0
        if similarity >= threshold:
            return True
    return False


def _principle_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams for principle similarity comparison."""
    return _char_ngrams(text, n)


def process_ghost(
    ghost: Ghost,
    trajectories: list[GhostTrajectory],
    all_ghosts: dict[str, Ghost],
    metabolism_rate: float = 0.5,
    existing_principles: list[str] | None = None,
) -> tuple[Ghost, GhostTrajectory, list[str]]:
    """Process a new ghost through the full pipeline.

    Steps:
    1. Assign ghost to a trajectory (create new if needed)
    2. Add ghost to trajectory (update cumulative PE)
    3. Check if trajectory should fire
    4. If firing: sublimate → extract principle (with dedup + sanitize)

    Returns:
    - Updated ghost (with trajectory_id set)
    - Updated trajectory
    - List of fired principle strings (empty if no firing)
    """
    fired_principles: list[str] = []

    # 1. Assign to trajectory
    trajectory, is_new_trajectory = assign_ghost_to_trajectory(ghost, trajectories)

    # 2. Add ghost to trajectory
    trajectory = add_ghost_to_trajectory(ghost, trajectory)
    ghost.trajectory_id = trajectory.id

    # 3. Check firing
    if should_fire(trajectory, metabolism_rate):
        trajectory.fired = True
        trajectory.fired_at = _now_str()
        trajectory.status = "fired"

        # 4. Sublimate
        all_ghosts[ghost.id] = ghost  # ensure latest ghost is in dict
        principle = sublimate(trajectory, all_ghosts)

        # 4a. Sanitize markdown noise
        principle = _sanitize_principle(principle)

        # 4b. Quality gate — reject task-specific principles
        if principle and not is_principle_generalizable(principle):
            principle = ""  # suppress non-generalizable

        # 4c. Dedup against existing principles
        all_existing = existing_principles or []
        if principle and _is_duplicate_principle(principle, all_existing):
            principle = ""  # suppress duplicate

        trajectory.sublimated_principle = principle

        if principle:
            fired_principles.append(principle)

    return ghost, trajectory, fired_principles


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")


def _make_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def ghost_to_dict(ghost: Ghost) -> dict:
    d = asdict(ghost)
    return d


def ghost_from_dict(data: dict) -> Ghost:
    return Ghost(
        id=data.get("id", ""),
        created_at=data.get("created_at", ""),
        rejected_output=data.get("rejected_output", ""),
        predicted_outcome=data.get("predicted_outcome", ""),
        interference=float(data.get("interference", 0.0)),
        actual_outcome=data.get("actual_outcome", ""),
        prediction_error=float(data.get("prediction_error", 0.0)),
        origin=data.get("origin", "rejected"),
        task_scope=data.get("task_scope", ""),
        tags=list(data.get("tags", [])),
        source_turn_id=data.get("source_turn_id", ""),
        trajectory_id=data.get("trajectory_id", ""),
    )


def trajectory_to_dict(trajectory: GhostTrajectory) -> dict:
    d = asdict(trajectory)
    return d


def trajectory_from_dict(data: dict) -> GhostTrajectory:
    return GhostTrajectory(
        id=data.get("id", ""),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        theme=data.get("theme", ""),
        ghost_ids=list(data.get("ghost_ids", [])),
        cumulative_pe=float(data.get("cumulative_pe", 0.0)),
        firing_threshold=float(data.get("firing_threshold", 1.0)),
        fired=bool(data.get("fired", False)),
        fired_at=data.get("fired_at", ""),
        sublimated_principle=data.get("sublimated_principle", ""),
        source_ghost_count=int(data.get("source_ghost_count", 0)),
        scopes=list(data.get("scopes", [])),
        origin_mix=dict(data.get("origin_mix", {})),
        status=data.get("status", "open"),
    )
