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
class Tension:
    """A contradiction pair between two rules/principles.

    When two rules point in opposite directions, neither is wrong —
    the contradiction reveals the BOUNDARY CONDITION: when to apply A
    vs B.  Extracting this boundary is the closest thing to learning
    human *judgment* rather than human *rules*.

    Detection is server-side (keyword overlap + directive opposition).
    Boundary extraction is client-LLM-side (inverted architecture).
    """
    id: str
    rule_a_id: str                          # source rule/principle ID
    rule_a_text: str                        # human-readable statement
    rule_b_id: str
    rule_b_text: str
    boundary: str = ""                      # "when A, when B" — the decision function
    signal: str = ""                        # what observable cue triggers the switch
    evidence_a: list[str] = field(default_factory=list)  # turn IDs where A was correct
    evidence_b: list[str] = field(default_factory=list)  # turn IDs where B was correct
    scopes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"                  # active / resolved / superseded


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
class Policy:
    """A deep, interpretable knowledge unit derived from accumulated rules/ghosts.

    Unlike rules (literal, single-use) or laws (constraint-only),
    a policy carries enough context to enable analogy, extension,
    inversion, and boundary detection in novel situations.

    Maturity lifecycle: rules accumulate → cluster detected → policy proposed → user approves.
    Once a policy is active, its source rules become dormant (not deleted).
    """
    id: str
    title: str                          # short name e.g. "理解が行動に先行する"
    core: str                           # one-line essence
    why: str                            # reasoning / motivation
    analogy: str = ""                   # how to extend by analogy
    opposite: str = ""                  # when NOT to apply / inverse
    limits: str = ""                    # boundary conditions
    source_rule_ids: list[str] = field(default_factory=list)
    source_ghost_ids: list[str] = field(default_factory=list)
    source_law_ids: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    evidence_count: int = 0             # how many raw data points support this
    maturity: str = "proposed"          # "proposed" | "active" | "superseded"
    created_at: str = ""
    updated_at: str = ""
    approved_by: str = ""               # "user" | "auto" | ""


@dataclass(slots=True)
class CuriositySignal:
    """A question detected by the client LLM from user messages.

    The Curiosity Layer is the third learning layer — it operates upstream
    of corrections (Surface Layer) and anger (Ghost Layer) by detecting
    when users ask questions, classifying those questions, and tracking
    knowledge gaps.

    Causal chain intercepted: question → repetition → resignation → anger → abandonment.

    The client LLM (not the server) handles detection and classification.
    The server handles persistence and clustering.
    """
    id: str
    created_at: str

    # The question extracted by the client LLM
    question_text: str = ""

    # Classification by client LLM
    # "knowledge_gap":          user doesn't know → teach
    # "judgment_uncertainty":   user knows but can't decide → provide criteria
    # "confirmation_seeking":   user knows but wants reassurance → confirm facts
    question_type: str = ""

    # "self":  user is asking for themselves (「わかりやすく教えて」)
    # "other": user is asking to translate for someone else (「わかりやすくまとめて」)
    target: str = "self"

    # Context
    source_turn_id: str = ""
    task_scope: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Client LLM's classification confidence (0.0-1.0)
    confidence: float = 0.0

    # Cluster linkage (filled when assigned)
    cluster_id: str = ""


@dataclass(slots=True)
class KnowledgeGapCluster:
    """A cluster of related questions in the same scope.

    Analogous to GhostTrajectory: individual questions are noise,
    but repeated questions in the same scope = knowledge gap signal.

    Escalation score rises with repeat count and signal density.
    When escalation is high, the client LLM should intervene proactively.
    """
    id: str
    created_at: str
    updated_at: str

    # Cluster definition
    scope: str = ""
    theme_keywords: list[str] = field(default_factory=list)
    dominant_type: str = ""          # most frequent question_type

    # Signal tracking
    signal_ids: list[str] = field(default_factory=list)
    signal_count: int = 0
    repeat_count: int = 0            # how many times similar questions recur

    # Escalation (0.0=calm, 1.0=about to give up)
    escalation_score: float = 0.0
    gap_strength: float = 0.0

    # Lifecycle
    status: str = "open"             # "open" | "resolved" | "escalated"
    resolved_at: str = ""
    scopes: list[str] = field(default_factory=list)


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
