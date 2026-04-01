from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConversationTurn:
    id: str
    recorded_at: str
    task_scope: str = ""
    user_message: str = ""
    assistant_message: str = ""
    user_feedback: str = ""
    extracted_corrections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Auto-scored from user reaction. None = not yet scored.
    reaction_score: float | None = None
    # True if guidance was injected before the assistant_message was generated.
    guidance_applied: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class RuleContext:
    kind: str
    value: str
    evidence_count: int = 0
    reaction_min: float | None = None
    reaction_max: float | None = None
    last_seen_at: str = ""
    utility_score: float = 0.0
    confidence_score: float = 0.0
    strong_signal_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass(slots=True)
class LatentContext:
    id: str
    scope: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    prototype_text: str = ""
    evidence_count: float = 0.0
    support_score: float = 0.0
    expected_gain: float = 0.0
    confidence_score: float = 0.0
    prior_weight: float = 0.0
    posterior_mass: float = 0.0
    strong_signal_count: float = 0.0
    success_mass: float = 0.0
    failure_mass: float = 0.0
    last_seen_at: str = ""


@dataclass(slots=True)
class LatentTransition:
    id: str
    from_signature: str
    to_signature: str
    from_scope: str = ""
    to_scope: str = ""
    from_tags: list[str] = field(default_factory=list)
    to_tags: list[str] = field(default_factory=list)
    from_keywords: list[str] = field(default_factory=list)
    to_keywords: list[str] = field(default_factory=list)
    evidence_count: float = 0.0
    success_weight: float = 0.0
    failure_weight: float = 0.0
    confidence_score: float = 0.0
    prediction_hit_count: float = 0.0
    prediction_miss_count: float = 0.0
    forecast_score: float = 0.0
    last_seen_at: str = ""


@dataclass(slots=True)
class PreferenceRule:
    id: str
    statement: str
    normalized_statement: str
    instruction: str
    status: str
    evidence_count: int
    first_recorded_at: str
    last_recorded_at: str
    applies_to_scope: str = ""
    applies_when_tags: list[str] = field(default_factory=list)
    negative_conditions: list[str] = field(default_factory=list)
    priority: int = 1
    version: int = 1
    expires_at: str = ""
    tags: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    contexts: list[RuleContext] = field(default_factory=list)
    latent_contexts: list[LatentContext] = field(default_factory=list)
    context_mode: str = "local"
    support_score: float = 0.0
    expected_gain: float = 0.0
    confidence_score: float = 0.0
    strong_signal_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    distinct_scope_count: int = 0
    distinct_tag_count: int = 0


@dataclass(slots=True)
class CorrectionRecord:
    recorded_at: str
    decision_override: str = ""
    correction_note: str = ""
    reuse_note: str = ""
    reason: str = ""
    scope: str = ""
    bad_output: str = ""
    revised_output: str = ""
    tool_used: str = ""
    source_user: str = ""
    accepted: bool = True


@dataclass(slots=True)
class TrainingExample:
    updated_at: str
    format: str = "chat"
    system_message: str = ""
    user_message: str = ""
    prompt: str = ""
    draft_output: str = ""
    rejected_output: str = ""
    accepted_output: str = ""
    feedback: str = ""
    accepted: bool = False
    model_id: str = ""
    policy_version: str = ""
    accepted_by: str = ""
    tags: list[str] = field(default_factory=list)
    temperature: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class Meaning:
    id: str
    principle: str
    normalized_principle: str
    summary: str
    source_rule_ids: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    strength: int = 0
    cross_scope_count: int = 0
    confidence: float = 0.0
    first_seen_at: str = ""
    last_seen_at: str = ""
    personal_settings_overlap: list[str] = field(default_factory=list)
    status: str = "active"


@dataclass(slots=True)
class Principle:
    id: str
    declaration: str              # 人格宣言テキスト
    normalized_declaration: str
    source_meaning_ids: list[str] = field(default_factory=list)
    source_rule_count: int = 0    # 全ソースルール数（意味経由）
    depth: int = 3                # 抽象レベル (1=rule, 2=meaning, 3=principle)
    scopes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    first_seen_at: str = ""
    last_seen_at: str = ""
    personal_settings_overlap: list[str] = field(default_factory=list)
    status: str = "active"


@dataclass(slots=True)
class Ghost:
    """A rejected AI proposal that persists as a counterfactual memory.

    When the AI proposes something and the user rejects or corrects it,
    the rejected option isn't discarded — it's stored as a ghost.
    The ghost carries the AI's predicted outcome and, when the actual
    outcome is later observed, the prediction error is computed.

    Trajectories of connected ghosts (same interference theme) eventually
    "fire" when cumulative prediction error crosses a threshold, triggering
    sublimation into an unlanguaged principle that feeds back autonomously.
    """
    id: str
    created_at: str

    # The rejected proposal
    rejected_output: str = ""          # what the AI proposed that was rejected

    # The prediction at rejection time
    predicted_outcome: str = ""        # AI's predicted user response (text)

    # Hesitation signal (proxy for quantum interference)
    # 0.0 = no hesitation, 1.0 = maximum hesitation
    interference: float = 0.0

    # Actual outcome (filled in when observed)
    actual_outcome: str = ""           # observed user response
    prediction_error: float = 0.0     # divergence between predicted and actual

    # Origin classification — scolded has highest signal quality
    # "rejected":  AI proposed, user chose differently (weak signal)
    # "corrected": AI output needed explicit correction (medium signal)
    # "scolded":   AI output caused frustration/anger (strongest signal)
    origin: str = "rejected"

    # Context
    task_scope: str = ""
    tags: list[str] = field(default_factory=list)
    source_turn_id: str = ""           # which ConversationTurn created this

    # Trajectory linkage (filled when assigned)
    trajectory_id: str = ""


@dataclass(slots=True)
class GhostTrajectory:
    """A sequence of related ghosts converging on the same interference theme.

    Trajectories form by clustering ghosts that share a common topic of
    conflict — the AI's tendency to misjudge the same thing repeatedly.

    When cumulative prediction error crosses the firing threshold, the
    trajectory fires. Firing triggers sublimation: extracting a principle
    from the pattern of rejections without requiring further human correction.
    This is the autonomous learning loop.
    """
    id: str
    created_at: str
    updated_at: str

    # Inferred theme of this interference pattern
    theme: str = ""

    # Ghost IDs ordered by creation time
    ghost_ids: list[str] = field(default_factory=list)

    # Accumulated prediction error (drives toward firing threshold)
    cumulative_pe: float = 0.0

    # Adaptive threshold (starts at 1.0, adjusts with metabolism_rate)
    firing_threshold: float = 1.0

    # Firing state
    fired: bool = False
    fired_at: str = ""

    # Sublimated principle (extracted autonomously when fired)
    sublimated_principle: str = ""

    # Metadata
    source_ghost_count: int = 0
    scopes: list[str] = field(default_factory=list)
    origin_mix: dict = field(default_factory=dict)  # {"scolded": N, ...}
    status: str = "open"  # "open" | "fired" | "exhausted"


@dataclass(slots=True)
class EpisodeRecord:
    id: str
    timestamp: str
    title: str
    issuer: str = ""
    task_type: str = "generic"
    profile_name: str = ""
    source_text: str = ""
    company_profile: dict = field(default_factory=dict)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    last_corrected_at: str = ""
    output: dict = field(default_factory=dict)
    training_example: TrainingExample | None = None
    metadata: dict = field(default_factory=dict)
