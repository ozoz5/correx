# CORREX

**Every correction you make to Claude becomes a rule. Rules compound. Your AI grows.**

CORREX is an MCP server that captures your feedback and turns it into persistent, injectable memory.
The more you use it, the more Claude thinks like you.

> Powered by the **Engram engine** — traces behavioral patterns from real interactions, promotes them to rules, and injects them before Claude responds.

---

## The problem

Every time you correct Claude — *"don't do it that way"*, *"be more concise"*, *"show the code first"* — that correction vanishes.

Tomorrow, Claude starts over. Same mistakes. Same friction.

CORREX fixes this.

---

## How it works

```
You correct Claude
        ↓
CORREX records the correction
        ↓
Engram detects the pattern (across multiple sessions)
        ↓
Pattern is promoted to a Rule
        ↓
Next session: Rule is injected before Claude responds
        ↓
Claude already knows what you want
```

After weeks of use, your Claude is not the same as anyone else's Claude.

---

## It actually works

This is a live system. These numbers are from real usage, not a demo.

**Memory after a few weeks of use:**

```
Rules:      100 total  (44 promoted, 56 still learning)
Turns:       77 recorded corrections
Meanings:    25 cross-scope patterns extracted
```

**Sample promoted rules (extracted from real corrections):**

```
[proposal_summary]  提案書要約では顧客の具体的な業務名を必ず含める
                    evidence=2  confidence=0.71

[architecture]      目的関数を固定値として扱うな。ユーザーの状態でゴールポストが動く。
                    evidence=1  confidence=0.60

[coding]            テストが通らないコードをコミットするな
                    evidence=3  confidence=0.88
```

**Measured quality improvement (before vs. after rule injection):**

| Task | Baseline | With rules | Delta |
|---|---|---|---|
| Proposal summary | 0.30 | 0.85 | **+55%** |
| LLM A/B simulation | 0.37 | 0.55 | **+18%** |
| Commercialization proposal | 0.62 | 0.75 | **+13%** |

**Test suite:**

```
$ python -m pytest tests/ -q
......................................................................
..................
57 passed in 2.03s
```

---

## The Engram engine

Engram is the memory layer inside CORREX. It operates in three learning layers:

| Layer | Signal | What it does |
|---|---|---|
| **Surface** (corrections) | User says "wrong" | Extracts rules, promotes to meanings/principles |
| **Ghost** (anger) | User rejects AI proposal | Clusters rejections, autonomously extracts principles |
| **Curiosity** (questions) | User asks questions | Tracks knowledge gaps, warns before frustration |

Rules promote only when they appear consistently across sessions.
Low-confidence rules are automatically demoted.
Contradicting rules are auto-resolved.

Engram also builds a **personality profile** (6 dimensions):
- `metabolism_rate` — how fast you change your mind
- `reward_pattern` — what makes you say "yes"
- `avoidance_pattern` — what makes you say "wrong"
- `digestibility` — abstract vs. concrete preference
- `curiosity_level` — how often you ask exploratory questions
- `objective_drift` — whether your goals have shifted

---

## Quickstart

### Install

```bash
git clone https://github.com/ozoz5/correx
cd correx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"
```

### Add to Claude Code

```bash
claude mcp add correx \
  -s user \
  -- /path/to/correx/.venv/bin/python \
  -m correx \
  --memory-dir ~/.correx \
  --transport stdio
```

### Add to Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "correx": {
      "command": "/path/to/correx/.venv/bin/python",
      "args": [
        "-m", "correx",
        "--memory-dir", "/Users/you/.correx",
        "--transport", "stdio"
      ]
    }
  }
}
```

Restart Claude. CORREX tools appear automatically.

### Enable the auto-learning loop

Copy the template into your project's `CLAUDE.md`:

```bash
cat /path/to/correx/CLAUDE_TEMPLATE.md >> .claude/CLAUDE.md
```

This instructs Claude to automatically call CORREX tools when you correct it, ask questions, or complete tasks.

---

## The core loop

```python
# At the start of any task
build_guidance_context(task_title="Write a proposal", task_scope="proposal")

# After Claude responds and you correct it
save_conversation_turn(
    task_scope="proposal",
    user_message="Write the summary",
    assistant_message="Here is a dense summary...",
    user_feedback="Too abstract. Use the client's actual business name.",
    extracted_corrections=["Always include the client's business name in summaries"],
    guidance_applied=True,
)

# Rules promote automatically. Next time: Claude already knows.
```

---

## MCP tools

| Tool | What it does |
|---|---|
| `build_guidance_context` | Injects your accumulated rules into Claude's context |
| `save_conversation_turn` | Records a correction or approval |
| `rebuild_preference_rules` | Re-scans full history and re-promotes rules |
| `synthesize_meanings` | Extracts deeper patterns from rule clusters |
| `synthesize_principles` | Distills principles from meanings |
| `get_personality_profile` | Shows behavioral profile + self-critique proposals |
| `synthesize_rules` | Generates rule hypotheses from success/failure patterns |
| `record_growth` | Measures before/after quality improvement |
| `save_curiosity_signal` | Records a user question (classified by client LLM) |
| `get_cognitive_map` | Shows knowledge gap map by scope |
| `save_ghost` | Records a rejected AI proposal for autonomous learning |
| `get_ghost_principles` | Returns autonomously extracted principles |

---

## Dashboard

A visual dashboard companion exists as a separate project (Next.js).
It is not included in this repository.

The overview shows a **living plant** that grows as your AI learns:

- Leaves → rules
- Flowers → meanings
- Fruits → principles

The plant grows slowly. Like your AI.

> Dashboard open-source release: planned.

---

## Architecture

```
correx/
  src/correx/
    mcp_server.py          # MCP tool definitions
    service.py             # Core service layer
    history_store.py       # Persistence + I/O orchestration
    rule_builder.py        # Pure rule construction logic (no I/O)
    memory_manager.py      # Conflict resolution, self-correction
    meaning_synthesis.py   # Engram: rules → meanings → principles
    personality_layer.py   # Behavioral profiling
    llm_scorer.py          # Reaction scoring (Anthropic API / rule-based fallback)
  tests/                   # 57 tests passing
```

All data stored as JSON in `~/.correx/`. No database required.

---

## Optional: LoRA training

CORREX can export your correction history as training data and trigger LoRA fine-tuning on Apple Silicon.

```bash
pip install -e ".[train]"

python3 scripts/auto_train.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --memory-dir ~/.correx \
  --output-dir ./training_artifacts
```

---

## Philosophy

Most AI personalization is about **data** — what the AI knows about you.
CORREX is about **behavior** — how the AI thinks with you.

The difference:
- Knowing you're vegetarian
- vs. knowing you hate when someone buries the conclusion

AGI is general intelligence for everyone.
CORREX is specific intelligence for one person.

---

## Status

- ✅ Working prototype
- ✅ 90 tests passing
- ✅ Running in production (single-user, local JSON)
- 🔄 Multi-user / hosted version in progress

---

## License

MIT
