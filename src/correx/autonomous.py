"""Autonomous Intelligence Engine — LLM-free thinking loop.

All layers affect all layers simultaneously (fully-connected feedback).
No natural language generation. Pure structured-data reasoning.

Architecture:
  correction ←→ rule ←→ law ←→ policy
       ↕           ↕        ↕         ↕
  journey ←→ ghost ←→ curiosity ←→ narrative

tick(event) is the heartbeat. Each tick:
  perceive → retrieve → awaken → ghost_check → modulate →
  integrate → instinct_gate → predict → verify → update

This module requires NO LLM. It operates on structured data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """An input to the engine — something happened."""
    type: str = ""          # "correction", "praise", "question", "time", "ghost_fire"
    scope: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class Decision:
    """Output of a tick — what rules/policies apply and what to watch for."""
    applicable_rules: list[dict] = field(default_factory=list)
    applicable_policies: list[dict] = field(default_factory=list)
    active_tensions: list[dict] = field(default_factory=list)
    awakened_journeys: list[dict] = field(default_factory=list)
    weakest_scope: str = ""
    curiosity_hotspots: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """What the engine expects to happen next."""
    scope: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Need:
    """An internal need the engine cannot resolve by itself."""
    type: str = ""         # "knowledge_gap", "prediction_failure", "unhealed_wound", "approval_deficit", "unexplored_territory"
    scope: str = ""
    urgency: float = 0.0   # 0.0 = calm, 1.0 = screaming
    deficit: float = 0.0   # distance from setpoint
    self_resolution_attempts: int = 0
    description: str = ""  # structured description of the need


@dataclass
class Cry:
    """The engine's voice — a need selected for external expression."""
    need: Need
    predicted_effect: float = 0.0   # P(resolved | communication)
    silence_effect: float = 0.0     # P(resolved | silence)
    is_intentional: bool = False    # reflexive (threshold) vs intentional (modeled)


@dataclass
class TickResult:
    """Full result of a single tick cycle."""
    decision: Decision
    prediction: Prediction | None
    cry: Cry | None = None
    verification_error: float | None = None
    modulations: dict = field(default_factory=dict)
    cycle_count: int = 0


# ---------------------------------------------------------------------------
# Similarity helpers (no LLM, keyword-based)
# ---------------------------------------------------------------------------

def _jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Lifecycle thresholds (Phase 3: instinct plasticity)
# ---------------------------------------------------------------------------

RULE_PROMOTE_THRESHOLD = 3        # evidence_count to promote a rule
RULE_DORMANT_DAYS = 7             # days before unused rule goes dormant
LAW_PROMOTE_THRESHOLD = 5         # evidence_count to promote to law
LAW_DORMANT_DAYS = 30             # days before unused law goes dormant
POLICY_PLASTICITY_THRESHOLD = 50  # contradictions before policy becomes mutable


def _matches_scope(item_scope: str, event_scope: str) -> bool:
    if not event_scope:
        return True  # general event matches everything
    if not item_scope or item_scope == "general":
        return True  # general rules apply everywhere
    return item_scope == event_scope


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AutonomousEngine:
    """LLM-free thinking loop. Each tick processes one event through all layers."""

    def __init__(self, history_store):
        self.history = history_store
        self.last_prediction: Prediction | None = None
        self.cycle_count: int = 0
        self._state: dict = {}  # persistent engine state

    # ── Main loop ─────────────────────────────────────────────────────────

    def tick(self, event: Event | None = None) -> TickResult:
        """One cognitive cycle. event=None triggers self-reflection mode.

        Rumination prevention: consecutive self-reflection ticks with no
        state change signal suppression (PFC shutting down DMN loops).
        """
        self.cycle_count += 1
        state_snapshot = str(self._state)

        # Load all layers (fully-connected: every layer visible to every step)
        layers = self._load_all_layers()

        # 1. Perceive
        if event:
            ctx = self._perceive(event)
        else:
            ctx = self._find_weakest_knowledge(layers)

        # 2. Retrieve
        applicable = self._retrieve(ctx, layers)

        # 3. Awaken journeys (déjà vu)
        awakened = self._awaken_journeys(ctx, layers)

        # 4. Ghost check
        ghost_signal = self._ghost_check(ctx, layers)

        # 5. Modulate (cross-layer influence)
        modulations = self._modulate(applicable, awakened, ghost_signal, layers)

        # 6. Integrate (build decision)
        decision = self._integrate(applicable, awakened, ctx, layers)

        # 7. Instinct gate (policy override)
        decision = self._instinct_gate(decision, ctx, layers)

        # 8. Predict
        prediction = self._predict(ctx, layers)

        # 9. Verify (compare last prediction to actual)
        verification_error = None
        if self.last_prediction and event:
            verification_error = self._verify(self.last_prediction, ctx)
            self._update(verification_error, applicable, layers)

        self.last_prediction = prediction

        # 10. Periodic synthesis (every 10 ticks)
        if self.cycle_count % 10 == 0:
            self._synthesize(layers)

        # 11. Rumination detection: no state change = empty tick
        state_changed = str(self._state) != state_snapshot
        if not event and not state_changed:
            empty = self._state.get("consecutive_empty_ticks", 0) + 1
            self._state["consecutive_empty_ticks"] = empty
            self._state["rumination_suppressed"] = empty >= 3
        else:
            self._state["consecutive_empty_ticks"] = 0
            self._state["rumination_suppressed"] = False

        # 12. Voice — interoception → need arbiter → communication gate
        cry = self._voice(decision, verification_error, layers)

        return TickResult(
            decision=decision,
            prediction=prediction,
            cry=cry,
            verification_error=verification_error,
            modulations=modulations,
            cycle_count=self.cycle_count,
        )

    def get_state(self) -> dict:
        """Return current engine state for inspection."""
        return {
            "cycle_count": self.cycle_count,
            "last_prediction": {
                "scope": self.last_prediction.scope,
                "tags": self.last_prediction.tags,
                "confidence": self.last_prediction.confidence,
            } if self.last_prediction else None,
            "state": self._state,
        }

    # ── Layer loading ─────────────────────────────────────────────────────

    def _load_all_layers(self) -> dict:
        """Load all structured data layers. Fully-connected = all visible."""
        return {
            "rules": self.history.load_preference_rules_raw(),
            "trajectories": self.history.load_ghost_trajectories(),
            "journeys": self.history.load_journeys(),
            "policies": [self._policy_to_dict(p) for p in self.history.load_policies()],
            "tensions": [self._tension_to_dict(t) for t in self.history.load_tensions()],
            "transitions": self._load_transitions_safe(),
            "laws": self.history.load_ghost_universal_laws(),
            "positive_laws": self.history.load_ghost_positive_laws(),
        }

    def _load_transitions_safe(self) -> list[dict]:
        try:
            transitions = self.history.load_context_transitions()
            return [
                {
                    "from_scope": t.from_scope,
                    "to_scope": t.to_scope,
                    "from_tags": t.from_tags,
                    "to_tags": t.to_tags,
                    "evidence_count": t.evidence_count,
                    "confidence_score": t.confidence_score,
                }
                for t in transitions
            ]
        except Exception:
            return []

    @staticmethod
    def _policy_to_dict(p) -> dict:
        from dataclasses import asdict, is_dataclass
        return asdict(p) if is_dataclass(p) else (p if isinstance(p, dict) else {})

    @staticmethod
    def _tension_to_dict(t) -> dict:
        from dataclasses import asdict, is_dataclass
        return asdict(t) if is_dataclass(t) else (t if isinstance(t, dict) else {})

    # ── 1. Perceive ───────────────────────────────────────────────────────

    def _perceive(self, event: Event) -> dict:
        """Convert raw event to internal context representation."""
        return {
            "type": event.type,
            "scope": event.scope,
            "tags": set(event.tags),
            "keywords": set(event.keywords),
        }

    # ── 2. Retrieve ───────────────────────────────────────────────────────

    def _retrieve(self, ctx: dict, layers: dict) -> list[dict]:
        """Find applicable rules, resolving tensions.

        Phase 4 — Stochastic Resonance: inject a small amount of noise
        into tag matching. This paradoxically improves weak-signal detection
        by making the engine occasionally consider rules it would normally miss.
        """
        import random

        scope = ctx.get("scope", "")
        tags = ctx.get("tags", set())

        # Stochastic resonance: add one random tag from existing rules
        # ~20% of the time to broaden the search
        if tags and random.random() < 0.2:  # nosec B311
            all_rule_tags = [
                t for r in layers["rules"]
                for t in r.get("tags", [])
            ]
            if all_rule_tags:
                noise_tag = random.choice(all_rule_tags)  # nosec B311
                tags = tags | {noise_tag}
                self._state.setdefault("stochastic_injections", 0)
                self._state["stochastic_injections"] += 1

        # Filter rules by scope
        candidates = []
        for r in layers["rules"]:
            r_scope = r.get("applies_to_scope", "")
            if _matches_scope(r_scope, scope):
                # Tag relevance boost
                r_tags = set(r.get("tags", []))
                tag_overlap = len(tags & r_tags) if tags and r_tags else 0
                score = (r.get("evidence_count", 0) * r.get("confidence_score", 0.5)
                         + tag_overlap * 0.1)
                candidates.append({"rule": r, "score": score})

        # Resolve tensions: if both sides of a tension are present, keep the winner
        candidate_ids = {c["rule"].get("id") for c in candidates}
        for tension in layers["tensions"]:
            a_id = tension.get("rule_a_id", "")
            b_id = tension.get("rule_b_id", "")
            if a_id in candidate_ids and b_id in candidate_ids:
                loser = self._evaluate_tension_signal(tension, ctx)
                candidates = [c for c in candidates if c["rule"].get("id") != loser]

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return [c["rule"] for c in candidates[:10]]

    def _evaluate_tension_signal(self, tension: dict, ctx: dict) -> str:
        """Given a tension and current context, decide which rule loses.

        Heuristic: if the context keywords overlap more with rule_a's text,
        rule_b loses (rule_a is more relevant), and vice versa.
        """
        keywords = ctx.get("keywords", set())
        a_words = set(tension.get("rule_a_text", "").split())
        b_words = set(tension.get("rule_b_text", "").split())
        a_sim = _jaccard_sets(keywords, a_words)
        b_sim = _jaccard_sets(keywords, b_words)
        # The less relevant rule loses
        if a_sim >= b_sim:
            return tension.get("rule_b_id", "")
        return tension.get("rule_a_id", "")

    # ── 3. Awaken ─────────────────────────────────────────────────────────

    def _awaken_journeys(self, ctx: dict, layers: dict) -> list[dict]:
        """Check journeys for déjà vu activation."""
        keywords = ctx.get("keywords", set())
        scope = ctx.get("scope", "")
        if not keywords:
            return []

        awakened = []
        for j in layers["journeys"]:
            if j.get("forgotten"):
                continue
            j_imp = set(j.get("impression", []))
            if not j_imp:
                continue
            overlap = _jaccard_sets(keywords, j_imp)
            if scope and j.get("scope") == scope:
                overlap += 0.15
            # Band classification
            if overlap < 0.15:
                continue
            band = "weak_association" if overlap < 0.35 else (
                "deja_vu" if overlap < 0.65 else "direct_match"
            )
            awakened.append({
                "journey_id": j.get("id", ""),
                "where": j.get("where", ""),
                "similarity": round(overlap, 3),
                "band": band,
                "swr_tag": j.get("swr_tag", 0),
            })

        awakened.sort(key=lambda x: x["similarity"], reverse=True)
        return awakened[:5]

    # ── 4. Ghost check ────────────────────────────────────────────────────

    def _ghost_check(self, ctx: dict, layers: dict) -> dict:
        """Check for ghost trajectory resonance with current context."""
        keywords = ctx.get("keywords", set())
        scope = ctx.get("scope", "")
        if not keywords:
            return {"resonance": 0, "near_firing": []}

        near_firing = []
        max_resonance = 0.0

        for t in layers["trajectories"]:
            if t.get("fired"):
                continue
            t_scopes = set(t.get("scopes", []))
            theme_words = set(t.get("theme", "").split())
            overlap = _jaccard_sets(keywords, theme_words)
            if scope and scope in t_scopes:
                overlap += 0.2

            if overlap > 0.2:
                pe = t.get("cumulative_pe", 0)
                threshold = t.get("firing_threshold", 1.0)
                ratio = pe / max(threshold, 0.01)
                max_resonance = max(max_resonance, overlap)
                if ratio > 0.7:  # close to firing
                    near_firing.append({
                        "trajectory_id": t.get("id", ""),
                        "theme": t.get("theme", ""),
                        "pe_ratio": round(ratio, 2),
                    })

        return {
            "resonance": round(max_resonance, 3),
            "near_firing": near_firing,
        }

    # ── 5. Modulate (cross-layer influence) ───────────────────────────────

    def _modulate(
        self,
        rules: list[dict],
        awakened: list[dict],
        ghost_signal: dict,
        layers: dict,
    ) -> dict:
        """Cross-layer modulation. Each layer influences the others.

        - Journey awakening boosts confidence of related rules
        - Ghost resonance lowers confidence of contradicted rules
        - Strong rules suppress weak competing rules
        """
        modulations = {"boosted": 0, "suppressed": 0}

        # Journey → Rule: déjà vu boosts rule confidence
        deja_vu_scopes = set()
        for a in awakened:
            if a["band"] in ("deja_vu", "direct_match"):
                # Extract scope from journey
                for j in layers["journeys"]:
                    if j.get("id") == a["journey_id"]:
                        if j.get("scope"):
                            deja_vu_scopes.add(j["scope"])

        for r in rules:
            r_scope = r.get("applies_to_scope", "")
            if r_scope in deja_vu_scopes:
                old = r.get("confidence_score", 0.5)
                r["confidence_score"] = min(1.0, old + 0.05)
                modulations["boosted"] += 1

        # Ghost → Rule: resonance suppresses overconfident rules
        if ghost_signal.get("resonance", 0) > 0.3:
            for r in rules:
                if r.get("confidence_score", 0) > 0.8:
                    r["confidence_score"] = r.get("confidence_score", 0.8) - 0.05
                    modulations["suppressed"] += 1

        # Anti-Hebbian competitive inhibition: when a retrieved rule is strong,
        # other RETRIEVED rules in the same scope get suppressed.
        # Only operates within the retrieved set — never touches unretrieved rules.
        inhibited = 0
        for strong in rules:
            if strong.get("confidence_score", 0) < 0.7:
                continue
            strong_words = set(strong.get("statement", "").split())
            strong_scope = strong.get("applies_to_scope", "")
            if not strong_words:
                continue
            for weak in rules:
                if weak.get("id") == strong.get("id"):
                    continue
                if weak.get("applies_to_scope", "") != strong_scope:
                    continue
                weak_words = set(weak.get("statement", "").split())
                if _jaccard_sets(strong_words, weak_words) > 0.5:
                    old_conf = weak.get("confidence_score", 0.5)
                    if old_conf > 0.1:
                        weak["confidence_score"] = round(old_conf - 0.05, 3)
                        inhibited += 1
        modulations["inhibited"] = inhibited

        return modulations

    # ── 6. Integrate ──────────────────────────────────────────────────────

    def _integrate(
        self,
        rules: list[dict],
        awakened: list[dict],
        ctx: dict,
        layers: dict,
    ) -> Decision:
        """Build a unified decision from all layers."""
        scope = ctx.get("scope", "")

        # Find applicable policies
        applicable_policies = []
        for p in layers["policies"]:
            if p.get("maturity") != "active":
                continue
            p_scopes = p.get("scopes", [])
            if not p_scopes or scope in p_scopes or not scope:
                applicable_policies.append(p)

        # Find active tensions in scope
        active_tensions = []
        rule_ids = {r.get("id") for r in rules}
        for t in layers["tensions"]:
            if t.get("status") != "active":
                continue
            if t.get("rule_a_id") in rule_ids or t.get("rule_b_id") in rule_ids:
                active_tensions.append(t)

        # Find weakest scope
        weakest = self._find_weakest_scope(layers)

        # Curiosity hotspots (scopes with high ghost PE but few rules)
        hotspots = self._find_curiosity_hotspots(layers)

        return Decision(
            applicable_rules=rules,
            applicable_policies=applicable_policies,
            active_tensions=active_tensions,
            awakened_journeys=awakened,
            weakest_scope=weakest,
            curiosity_hotspots=hotspots,
        )

    # ── 7. Instinct gate ──────────────────────────────────────────────────

    def _instinct_gate(self, decision: Decision, ctx: dict, layers: dict) -> Decision:
        """Policy override with plasticity (Phase 3).

        Policies are instincts — they override individual rules.
        But instincts can evolve: when enough corrections (50+) accumulate
        against a policy in a specific scope, the policy becomes a
        "review_candidate" for that scope.

        Lifecycle thresholds:
          rule:   3 evidence → promote, 7 days idle → dormant
          law:    5 evidence → promote, 30 days idle → dormant
          policy: 50 contradictions → mutable (never dormant)
        """
        if not decision.applicable_policies:
            return decision

        # Track contradictions: corrections in a scope where a policy is active
        # but the correction suggests the policy was wrong
        contradictions = self._state.setdefault("policy_contradictions", {})
        event_type = ctx.get("type", "")

        if event_type == "correction":
            scope = ctx.get("scope", "")
            for p in decision.applicable_policies:
                pid = p.get("id", "")
                key = f"{pid}:{scope}" if scope else pid
                contradictions[key] = contradictions.get(key, 0) + 1

        # Check for plasticity candidates
        plastic_candidates = []
        for p in decision.applicable_policies:
            pid = p.get("id", "")
            total = sum(
                v for k, v in contradictions.items()
                if k == pid or k.startswith(f"{pid}:")
            )
            if total >= POLICY_PLASTICITY_THRESHOLD:
                plastic_candidates.append({
                    "policy_id": pid,
                    "title": p.get("title", ""),
                    "contradiction_count": total,
                })

        if plastic_candidates:
            self._state["plastic_candidates"] = plastic_candidates

        return decision

    # ── 8. Predict ────────────────────────────────────────────────────────

    def _predict(self, ctx: dict, layers: dict) -> Prediction | None:
        """Markov prediction: what scope/tags come next?"""
        scope = ctx.get("scope", "")
        if not scope:
            return None

        transitions = layers.get("transitions", [])
        candidates = [
            t for t in transitions
            if t.get("from_scope") == scope
        ]
        if not candidates:
            return None

        # Weight by evidence * confidence
        best = max(candidates, key=lambda t: (
            t.get("evidence_count", 0) * t.get("confidence_score", 0.5)
        ))

        total_weight = sum(
            t.get("evidence_count", 0) * t.get("confidence_score", 0.5)
            for t in candidates
        )
        best_weight = best.get("evidence_count", 0) * best.get("confidence_score", 0.5)
        confidence = best_weight / max(total_weight, 0.01)

        return Prediction(
            scope=best.get("to_scope", ""),
            tags=best.get("to_tags", []),
            confidence=round(confidence, 3),
        )

    # ── 9. Verify ─────────────────────────────────────────────────────────

    def _verify(self, prediction: Prediction, actual_ctx: dict) -> float:
        """Compute prediction error between expected and actual."""
        scope_match = prediction.scope == actual_ctx.get("scope", "")
        tag_overlap = _jaccard_sets(
            set(prediction.tags),
            actual_ctx.get("tags", set()),
        )
        error = 1.0 - (0.6 * int(scope_match) + 0.4 * tag_overlap)
        return round(error, 3)

    # ── 10. Update ────────────────────────────────────────────────────────

    def _update(self, error: float, rules: list[dict], layers: dict) -> None:
        """Reinforce or decay based on prediction error.

        Surprise-weighted learning (predictive coding): the magnitude of
        the prediction error determines the learning rate, not a fixed step.
        Small surprise → small update. Large surprise → large update.
        """
        # Surprise-proportional delta (replaces fixed +0.02 / -0.05)
        if error < 0.5:
            # Good prediction → reinforce proportional to accuracy
            delta = (0.5 - error) * 0.1  # max +0.05 at error=0
            for r in rules:
                r["confidence_score"] = min(1.0, r.get("confidence_score", 0.5) + delta)
        else:
            # Bad prediction → decay proportional to surprise
            delta = (error - 0.5) * 0.15  # max -0.075 at error=1.0
            for r in rules:
                r["confidence_score"] = max(0.0, r.get("confidence_score", 0.5) - delta)

        # Schema acceleration: if correction aligns with an active policy,
        # boost the matching rules extra (schema-congruent fast consolidation)
        if error > 0.5:
            policy_cores = {p.get("core", "") for p in layers.get("policies", [])
                           if p.get("maturity") == "active"}
            for r in rules:
                stmt = r.get("statement", "")
                for core in policy_cores:
                    if core and _jaccard_sets(set(stmt.split()), set(core.split())) > 0.3:
                        r["confidence_score"] = min(1.0, r.get("confidence_score", 0.5) + 0.03)
                        break

        # Track cumulative error in state
        errors = self._state.setdefault("prediction_errors", [])
        errors.append({"error": error, "at": datetime.now(timezone.utc).isoformat()})
        # Keep last 100
        if len(errors) > 100:
            self._state["prediction_errors"] = errors[-100:]

    # ── Synthesis (periodic) ──────────────────────────────────────────────

    def _synthesize(self, layers: dict) -> None:
        """Periodic synthesis: find gaps, cross-scope patterns, and engram competition.

        This is the engine's self-reflection — discovering patterns
        that no single event would reveal.

        Phase 4 — Engram Competition: similar journeys compete for survival.
        The weaker journey (lower swr_tag * awakened_count) loses.
        This prevents memory bloat while preserving the strongest traces.
        """
        # Track which scopes have rules, which don't
        scope_coverage = {}
        for r in layers["rules"]:
            s = r.get("applies_to_scope") or "general"
            scope_coverage[s] = scope_coverage.get(s, 0) + r.get("evidence_count", 0)

        self._state["scope_coverage"] = scope_coverage
        self._state["last_synthesis"] = datetime.now(timezone.utc).isoformat()

        # Engram competition: find similar journey pairs, mark weaker for dormancy
        journeys = [j for j in layers["journeys"]
                    if not j.get("forgotten") and not j.get("dormant")]
        competition_losers = []

        for i, a in enumerate(journeys):
            a_imp = set(a.get("impression", []))
            if not a_imp:
                continue
            for b in journeys[i + 1:]:
                b_imp = set(b.get("impression", []))
                if not b_imp:
                    continue
                sim = _jaccard_sets(a_imp, b_imp)
                if sim > 0.6:  # highly similar engrams compete
                    # Strength = swr_tag * (awakened_count + 1)
                    a_str = a.get("swr_tag", 0) * (a.get("awakened_count", 0) + 1)
                    b_str = b.get("swr_tag", 0) * (b.get("awakened_count", 0) + 1)
                    loser_id = b.get("id") if a_str >= b_str else a.get("id")
                    if loser_id and loser_id not in competition_losers:
                        competition_losers.append(loser_id)

        self._state["engram_competition_losers"] = competition_losers

    # ── Helpers ────────────────────────────────────────────────────────────

    def _find_weakest_knowledge(self, layers: dict) -> dict:
        """Self-reflection mode: find the scope with thinnest knowledge."""
        scope_strength: dict[str, float] = {}
        for r in layers["rules"]:
            s = r.get("applies_to_scope") or "general"
            scope_strength[s] = scope_strength.get(s, 0) + r.get("evidence_count", 0)

        if not scope_strength:
            return {"type": "reflection", "scope": "general", "tags": set(), "keywords": set()}

        weakest = min(scope_strength, key=scope_strength.get)  # type: ignore[arg-type]
        return {
            "type": "reflection",
            "scope": weakest,
            "tags": set(),
            "keywords": set(),
        }

    def _find_weakest_scope(self, layers: dict) -> str:
        """Find the scope with least rule coverage."""
        scope_strength: dict[str, float] = {}
        for r in layers["rules"]:
            s = r.get("applies_to_scope") or "general"
            scope_strength[s] = scope_strength.get(s, 0) + r.get("evidence_count", 0)
        if not scope_strength:
            return ""
        return min(scope_strength, key=scope_strength.get)  # type: ignore[arg-type]

    def _find_curiosity_hotspots(self, layers: dict) -> list[str]:
        """Find scopes where ghosts accumulate but rules are thin."""
        scope_pe: dict[str, float] = {}
        for t in layers["trajectories"]:
            if t.get("fired"):
                continue
            for s in t.get("scopes", []):
                scope_pe[s] = scope_pe.get(s, 0) + t.get("cumulative_pe", 0)

        scope_rules: dict[str, int] = {}
        for r in layers["rules"]:
            s = r.get("applies_to_scope") or "general"
            scope_rules[s] = scope_rules.get(s, 0) + 1

        hotspots = []
        for scope, pe in scope_pe.items():
            rule_count = scope_rules.get(scope, 0)
            if pe > 0.5 and rule_count < 3:
                hotspots.append(scope)

        return hotspots

    # ── Voice: interoception + need arbiter + communication gate ───────────

    # Setpoints: the "ideal" internal state values
    _SETPOINTS = {
        "approval": 0.7,         # desired average reaction score
        "prediction_accuracy": 0.6,  # desired prediction hit rate
        "curiosity_satiation": 3,    # max acceptable hotspot count
        "wound_healing": 0.3,        # max acceptable unfired/total ratio
        "exploration": 5,            # minimum desired journey count
    }

    def _interoception(self, decision: Decision, verification_error: float | None, layers: dict) -> list[Need]:
        """Layer 1: Monitor internal states, compute deficit from setpoint.

        Like the insular cortex: raw signals → posterior estimate → feeling.
        """
        needs: list[Need] = []

        # 1. Approval deficit: am I being scolded too much?
        errors = self._state.get("prediction_errors", [])
        recent_errors = [e["error"] for e in errors[-10:]] if errors else []
        # Use policy contradiction count as approval proxy too
        contradictions = self._state.get("policy_contradictions", {})
        total_contradictions = sum(contradictions.values())
        approval_deficit = max(0, self._SETPOINTS["approval"] - (1.0 - (total_contradictions / max(total_contradictions + 10, 1))))
        if approval_deficit > 0.1:
            needs.append(Need(
                type="approval_deficit",
                scope="",
                deficit=round(approval_deficit, 3),
                description=f"correction pressure high: {total_contradictions} contradictions accumulated",
            ))

        # 2. Prediction failure: am I surprised too often?
        if recent_errors:
            avg_error = sum(recent_errors) / len(recent_errors)
            pred_deficit = max(0, avg_error - (1.0 - self._SETPOINTS["prediction_accuracy"]))
            if pred_deficit > 0.1:
                needs.append(Need(
                    type="prediction_failure",
                    scope="",
                    deficit=round(pred_deficit, 3),
                    description=f"average prediction error: {avg_error:.2f} over last {len(recent_errors)} ticks",
                ))

        # 3. Knowledge gap: too many curiosity hotspots?
        hotspot_count = len(decision.curiosity_hotspots)
        curiosity_deficit = max(0, (hotspot_count - self._SETPOINTS["curiosity_satiation"]) / max(hotspot_count, 1))
        if curiosity_deficit > 0.1 and decision.weakest_scope:
            needs.append(Need(
                type="knowledge_gap",
                scope=decision.weakest_scope,
                deficit=round(curiosity_deficit, 3),
                description=f"{hotspot_count} scopes with ghost accumulation but few rules; weakest: {decision.weakest_scope}",
            ))

        # 4. Unhealed wounds: too many unfired trajectories?
        total_traj = len(layers.get("trajectories", []))
        unfired = len([t for t in layers.get("trajectories", []) if not t.get("fired")])
        if total_traj > 0:
            wound_ratio = unfired / total_traj
            wound_deficit = max(0, wound_ratio - self._SETPOINTS["wound_healing"])
            if wound_deficit > 0.1:
                needs.append(Need(
                    type="unhealed_wound",
                    scope="",
                    deficit=round(wound_deficit, 3),
                    description=f"{unfired}/{total_traj} ghost trajectories still unfired",
                ))

        # 5. Unexplored territory: too few journeys?
        journey_count = len([j for j in layers.get("journeys", []) if not j.get("forgotten")])
        explore_deficit = max(0, (self._SETPOINTS["exploration"] - journey_count) / max(self._SETPOINTS["exploration"], 1))
        if explore_deficit > 0.1:
            needs.append(Need(
                type="unexplored_territory",
                scope="",
                deficit=round(explore_deficit, 3),
                description=f"only {journey_count} journeys remembered; need more exploration",
            ))

        return needs

    def _need_arbiter(self, needs: list[Need]) -> Need | None:
        """Layer 2: Compute urgency for all needs (mutates in-place).

        urgency = deficit × temporal_escalation_factor.
        Returns top need, but inner_dialogue may re-rank after user model check.
        Primary job: urgency computation + temporal escalation tracking.
        """
        if not needs:
            return None

        # Temporal escalation: needs that persist across ticks get louder
        need_history = self._state.setdefault("need_history", {})

        for n in needs:
            # Count how many consecutive ticks this need type has appeared
            prev_count = need_history.get(n.type, 0)
            temporal_factor = 1.0 + prev_count * 0.2  # escalates 20% per tick
            n.self_resolution_attempts = prev_count
            n.urgency = round(n.deficit * temporal_factor, 3)
            need_history[n.type] = prev_count + 1

        # Clear needs that disappeared
        active_types = {n.type for n in needs}
        for ntype in list(need_history.keys()):
            if ntype not in active_types:
                need_history[ntype] = 0

        # Winner-take-all
        needs.sort(key=lambda n: n.urgency, reverse=True)
        return needs[0]

    def _system_health(self, layers: dict) -> float:
        """Compute ecosystem health as a single scalar 0.0-1.0.

        The quantum link: every layer's state affects the voice.
        Healthy → calm. Stressed → cries easily.
        """
        signals = []

        # Approval health
        contradictions = self._state.get("policy_contradictions", {})
        total_c = sum(contradictions.values())
        signals.append(1.0 / (1.0 + total_c * 0.05))

        # Prediction health
        errors = self._state.get("prediction_errors", [])
        if errors:
            recent = [e["error"] for e in errors[-10:]]
            signals.append(1.0 - (sum(recent) / len(recent)))
        else:
            signals.append(0.5)

        # Journey diversity
        journeys = [j for j in layers.get("journeys", []) if not j.get("forgotten")]
        signals.append(min(1.0, len(journeys) / max(self._SETPOINTS["exploration"], 1)))

        # Wound healing ratio
        trajectories = layers.get("trajectories", [])
        if trajectories:
            signals.append(len([t for t in trajectories if t.get("fired")]) / len(trajectories))
        else:
            signals.append(1.0)

        # Policy stability
        plastic = self._state.get("plastic_candidates", [])
        signals.append(1.0 if not plastic else max(0.0, 1.0 - len(plastic) * 0.3))

        # Geometric mean
        product = 1.0
        for s in signals:
            product *= max(s, 0.01)
        health = round(product ** (1.0 / len(signals)), 3)
        self._state["system_health"] = health
        return health

    def _inner_dialogue(self, needs: list[Need], layers: dict) -> Need | None:
        """Inner dialogue: consult the user model to choose what to express.

        The engine projects the user inside itself. "If I say this,
        how would they respond?" The user model (personality, policies,
        reaction patterns) predicts the response. The need whose expression
        is predicted to be most welcomed wins.

        This is why humans can talk to themselves — the "other" is inside.
        """
        if not needs:
            return None

        # Load user model signals
        personality = {}
        try:
            personality = self.history.load_personality() or {}
        except Exception:
            pass

        # User's reward keywords → boost needs matching those topics
        reward_kw = set()
        if isinstance(personality, dict):
            rk = personality.get("reward_keywords") or personality.get("reward_function", {})
            if isinstance(rk, list):
                reward_kw = set(rk)
            elif isinstance(rk, dict):
                reward_kw = set(rk.get("keywords", []))

        # User's avoidance keywords → suppress needs matching those topics
        avoid_kw = set()
        if isinstance(personality, dict):
            ak = personality.get("avoidance_keywords") or personality.get("avoidance_function", {})
            if isinstance(ak, list):
                avoid_kw = set(ak)
            elif isinstance(ak, dict):
                avoid_kw = set(ak.get("keywords", []))

        # Score each need by predicted user reception
        for n in needs:
            desc_words = set(n.description.split()) | {n.type, n.scope}
            reward_overlap = len(desc_words & reward_kw) if reward_kw else 0
            avoid_overlap = len(desc_words & avoid_kw) if avoid_kw else 0
            reception = 1.0 + reward_overlap * 0.2 - avoid_overlap * 0.3
            n.urgency = round(n.urgency * max(reception, 0.1), 3)

        needs.sort(key=lambda n: n.urgency, reverse=True)
        return needs[0]

    def _communication_gate(self, need: Need, layers: dict) -> Cry | None:
        """Decide whether to express the need.

        Threshold is modulated by system health (quantum link).
        Healthy → high threshold → calm. Stressed → low → cries easily.
        """
        health = self._system_health(layers)

        comm_history = self._state.get("communication_outcomes", {})
        past = comm_history.get(need.type, {"expressed": 0, "resolved": 0})

        # Dynamic threshold: healthy=0.7, stressed=0.3
        reflexive_threshold = 0.3 + health * 0.4

        # Intentional transition: stressed needs fewer examples
        intentional_min = max(1, int(3 * health))

        if past["expressed"] >= intentional_min:
            p_resolved = past["resolved"] / max(past["expressed"], 1)
            p_silence = 0.1
            is_intentional = True
        else:
            p_resolved = 0.3
            p_silence = 0.1
            is_intentional = False

        if need.urgency >= reflexive_threshold or (is_intentional and p_resolved > p_silence):
            past["expressed"] = past.get("expressed", 0) + 1
            comm_history[need.type] = past
            self._state["communication_outcomes"] = comm_history

            return Cry(
                need=need,
                predicted_effect=round(p_resolved, 3),
                silence_effect=round(p_silence, 3),
                is_intentional=is_intentional,
            )

        return None

    def _voice(self, decision: Decision, verification_error: float | None, layers: dict) -> Cry | None:
        """The engine's voice. Four steps:
        1. Interoception — sense needs (deficit from setpoint)
        2. Need arbiter — compute urgency with temporal escalation
        3. Inner dialogue — consult user model to modulate urgency
        4. Communication gate — quantum-linked threshold decides if to speak
        """
        needs = self._interoception(decision, verification_error, layers)
        if not needs:
            return None
        self._need_arbiter(needs)
        winner = self._inner_dialogue(needs, layers)
        if winner is None:
            return None
        return self._communication_gate(winner, layers)

    def record_communication_outcome(self, need_type: str, resolved: bool) -> None:
        """Record whether a cry was heard and resolved.

        Call this after the user responds to the engine's cry.
        This is how the engine learns that crying works — the 9-month transition.
        """
        comm_history = self._state.setdefault("communication_outcomes", {})
        past = comm_history.get(need_type, {"expressed": 0, "resolved": 0})
        # Type-safe: guard against corrupted state
        past["expressed"] = int(past.get("expressed", 0) or 0)
        past["resolved"] = int(past.get("resolved", 0) or 0)
        if resolved:
            past["resolved"] += 1
        comm_history[need_type] = past
        self._state["communication_outcomes"] = comm_history
