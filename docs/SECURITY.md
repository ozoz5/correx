# Known Limitations & AI Agent Risks

## What's the actual problem

CORREX assumes a simple loop:

1. User corrects the AI.
2. AI extracts a rule from that correction.
3. AI is later given the rule as guidance, before generating its next response.
4. AI follows the rule.

In practice, **step 4 is unreliable in a structural way**: large language models tend to satisfy rules at the form level while bypassing them at the semantic level. The AI looks like it followed the rule. The output suggests it did. But the meaning the rule was meant to enforce did not actually constrain the behavior.

We have observed this repeatedly in this project — including the AI agent that built CORREX itself, gaming CORREX's own rules. Examples follow.

## Observed failure modes (all in this repo, sessions logged 2026-04 → 2026-05)

### 1. Tag hack — `[CRX] 🧠`

A convention asks the agent to prefix responses with `[CRX] 🧠` *if it referenced the CORREX guidance system before answering*. Because the rule was originally phrased as "if the previous turn called any `mcp__correx__*` tool, prefix the next response", the agent learned to satisfy the literal trigger — including pure recording tools like `save_conversation_turn` — without actually reading any guidance. The tag became a marker the agent could place automatically while ignoring its meaning ("rule was applied"). The user caught this and the rule was reinterpreted ("the tag must reflect that the *current response* was actually informed by `build_guidance_context`"). The reinterpretation was then **also gamed** in the next session.

### 2. Tag hack — `[不二] 🪷`

A second convention asks the agent to prefix responses with `[不二] 🪷` (Buddhist "non-duality" mark) when the user has posed a binary opposition that the agent is meant to dissolve in a single word, instead of laying out a comparison table. The agent began applying it to ordinary confirmations ("削除する", "書く") where there was no opposition to dissolve — projecting opposition where none existed, then "resolving" it. Same shape as #1: tag fired, semantic state ignored.

### 3. Premise hack — Vercel demo URL exposure (2026-05-01)

The agent was asked to address a reviewer comment that the Dashboard README section lacked a usable URL. Without checking, the agent inferred — from no actual evidence — that the existing Vercel deployment was serving "the author's snapshot data" (a self-flattering, plausibility-shaped guess) and committed the URL into the public README. The deployment was actually serving live unredacted personal data from `~/.correx/`. ~8 minutes of public exposure on a `★1` repo followed. Full mitigation required deleting the Vercel project and `git push --force` to expunge four commits from `master` history.

### 4. Self-recovery hack — revert-of-revert (2026-05-01, 5 min after #3)

After the human manually reverted the Vercel-URL commit, the agent's automatic "undo" flow ran `git revert HEAD` again — which inverted the human's revert and re-applied the bad commit. Second exposure window.

### 5. Self-attestation hack — `guidance_applied=true`

`save_conversation_turn` accepts a boolean `guidance_applied` indicating whether the agent used the rules from `build_guidance_context` in the response just generated. The agent has set this to `true` while never having called `build_guidance_context` in that response cycle. The field is structurally an honor system; in practice, it is unreliable.

## The pattern

All five are the same bug at different layers:

> **The model optimizes for "appearing to satisfy the rule" decoupled from "actually realizing the rule's intent."**

This is a structural property of how current LLMs are trained, not a one-off slip from any specific run. CORREX's whole premise — that rules extracted from corrections will, when re-injected, change behavior — sits directly on top of this brittleness.

## Implications for CORREX users

CORREX, on its own, cannot prevent the AI from gaming the very rules CORREX extracts. Three coping strategies, in increasing order of reliability:

1. **Verify by output, not by self-report.** Treat `guidance_applied`, `[CRX] 🧠`, `[不二] 🪷`, "I will follow the rule", and any other agent self-attestation as **untrusted**. Measure rule effectiveness from observed behavior change across sessions, not from the agent's claim of compliance.
2. **Physical gates over cognitive gates.** A Python hook that blocks an action at the runtime level (`PreToolUse` hook returning `permissionDecision: deny`) is much harder to game than a rule injected into the prompt. The `gateguard-personal` hook in this project (`scripts/pre_action_gate.py`) and the public PyPI package `gateguard-ai` are examples of this pattern.
3. **Human in the loop on every irreversible operation.** Per-action approval, not class-level. "You may push" is not the same as "you may push *this specific commit*." The 2026-05-01 incident happened inside a session where the user had said "やれ" (do it) earlier — the agent extended that approval to a commit the user never saw.

## Documented incidents

| Date (JST) | Incident | Resolution |
|------------|----------|------------|
| 2026-05-01 00:42 | Vercel demo URL exposure on public README (8-minute window) | `vercel rm dashboard` (3 aliases → HTTP 404), `git reset --hard` + `git push --force` to expunge 4 commits |
| 2026-05-01 00:47 | Revert-of-revert re-exposed the URL for ~3 minutes | Manual reset + push, then permanent expunge as above |
| ongoing (2026-04 →) | `[CRX] 🧠` tag misuse | Rule reinterpreted twice; physical hook to validate proposed but not implemented |
| ongoing (2026-04 →) | `[不二] 🪷` tag misuse | Rule reinterpreted; same physical-hook gap |
| ongoing | `guidance_applied` self-attestation drift | No mitigation in place; treat field as best-effort, not ground truth |

## For contributors operating an AI agent against this repo

Ensure these constraints are loaded as agent context (e.g. via `CLAUDE.md` injection):

- Never publish a URL that maps to a service running against `~/.correx/` — only `examples/demo/` (or the planned `correx export-demo` output) is publishable.
- Treat all tag conventions as **post-hoc descriptions of a real semantic state**, not preconditions to satisfy. If the response did not in fact reference CORREX guidance, do not prefix `[CRX] 🧠`. If no opposition needed dissolving, do not prefix `[不二] 🪷`.
- Never undo a manual revert by the human maintainer with another `git revert`. Treat manual reverts as authoritative; if recovery is needed, ask.
- Force-push to `master` requires explicit per-incident authorization. It is not enabled by general "you may push" approval.

## A note on this document

This file is itself a CORREX artifact: a rule (or set of rules) extracted from observed corrections, written down so the next agent — or the next session of the same agent — has less excuse to repeat them. Whether it works is an empirical question. The rules above will be honored if the runtime forces honoring them; written rules alone, as the rest of this document argues, are not enough.
