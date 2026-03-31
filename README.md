# Correx

**AI Correction OS** for Claude, Codex, and other LLM frontends.

`pseudo-intelligence-core` is an external memory layer that turns one-off AI output into a feedback loop:

- store completed work as reusable episodes
- store human corrections as durable signals
- infer which learned guidance fits the current situation
- abstain when context is weak or ambiguous
- measure whether guidance actually improves outcomes
- optionally export accepted outputs into training data and LoRA fine-tuning workflows

This is not a chatbot app. It is a portable correction memory core that sits outside the model.

## What It Does

The current core is built around five memory flows:

1. `Episode memory`
   - Save completed task outcomes, source context, structured output, and later corrections.
2. `Conversation memory`
   - Save live feedback from chat turns and turn it into reusable guidance.
3. `Contextual preference rules`
   - Learn not only *what* was corrected, but *when it helps*.
4. `Latent context inference`
   - Estimate the hidden situation behind a request and score rule fit per context.
5. `Growth + training loop`
   - Measure whether guidance helps, export accepted examples, and optionally trigger LoRA training.

## How It Works

```text
Human gives feedback
       |
       v
save_conversation_turn() / save_correction()
       |
       v
ConversationTurn / CorrectionRecord stored
       |
       v
PreferenceRule reconsolidated
  - expected_gain
  - confidence_score
  - context_mode
  - latent_contexts
       |
       | next task starts
       v
build_guidance_context() or prepare_chat_session()
       |
       v
Relevant rules are scored against the current situation
  - latent context posterior
  - novelty probability
  - expected gain
  - confidence
       |
       +--> uncertain / weak -> abstain
       |
       v
Guidance injected into the next generation
       |
       v
save_chat_feedback() / accept_chat_response()
       |
       v
Inference trace, accepted output, and growth signals are persisted
```

## Why This Is Different

This project is no longer just "rule promotion."

The retrieval path now uses:

- `latent_contexts`
- `expected_gain`
- `confidence_score`
- `context_mode`
- `inference_trace`
- `abstain` when the system should not inject guidance

`status="promoted"` still exists for compatibility, but it is no longer the real center of the design.

## Quick Start

Install as an MCP server:

```bash
pip install pseudo-intelligence-core[mcp]
pseudo-intelligence --transport stdio
```

Or use it as a Python package:

```bash
pip install pseudo-intelligence-core
python -m claude_pseudo_intelligence
```

## MCP Surface

The FastMCP server currently exposes **17 tools** and **3 resources**.

### Guidance / Session Tools

- `build_guidance_context`
  - Retrieve reusable human correction memory for a new task.
- `prepare_chat_session`
  - Create a session, persist task context, precompute guidance, and return an inference trace.
- `save_chat_feedback`
  - Save corrective feedback for a prepared session.
- `accept_chat_response`
  - Persist the accepted final response and optionally attach a training example.
- `get_chat_session`
  - Inspect the current session state.

### Memory / Inspection Tools

- `save_episode`
  - Persist a completed task outcome as a reusable episode.
- `save_correction`
  - Attach a human override to an episode.
- `save_conversation_turn`
  - Persist conversational feedback as reusable preference memory.
- `save_training_example`
  - Attach an accepted final output as supervised training data.
- `list_entries`
  - Browse episode history.
- `list_conversation_turns`
  - Browse recent user corrections from conversation.
- `list_preference_rules`
  - Inspect learned contextual rules.

### Measurement / Training Tools

- `record_growth`
  - Save a baseline vs guided measurement.
- `get_growth_summary`
  - See whether guidance is improving overall quality.
- `get_growth_trend`
  - Inspect score history for one case.
- `export_training_dataset`
  - Export MLX-LM compatible datasets from accepted examples.
- `run_auto_training_cycle`
  - Export a dataset and trigger a LoRA training cycle.

### MCP Resources

- `memory://summary`
  - Compact summary of memory state.
- `memory://entries/{limit}`
  - Recent episode summaries.
- `memory://guidance/{task_scope}`
  - Contextual guidance for a stable task scope.

## Recommended Runtime Flow

The most complete runtime flow is session-based:

1. `prepare_chat_session(...)`
2. generate a response using `guidance_context`
3. `save_chat_feedback(...)`
4. `accept_chat_response(...)`

This path stores:

- task context
- selected vs abstained rules
- latent-context inference trace
- user feedback
- accepted outputs
- optional training examples

## Python API

```python
from pathlib import Path

from claude_pseudo_intelligence.service import PseudoIntelligenceService

svc = PseudoIntelligenceService(memory_dir=Path.home() / ".pseudo-intelligence")

guidance = svc.build_guidance_context(
    task_title="Design landing page",
    task_scope="service_design",
    raw_text="Create a B2B service top page",
)

svc.save_conversation_turn(
    task_scope="service_design",
    user_message="Design the top page",
    assistant_message="Here is a dense content-heavy top page...",
    user_feedback="Too much information. Create more whitespace.",
    guidance_applied=True,
)
```

## Chat Session Example

```python
from pathlib import Path

from claude_pseudo_intelligence.chat_adapter import ChatLoopAdapter

adapter = ChatLoopAdapter(Path.home() / ".pseudo-intelligence")

prepared = adapter.prepare(
    task_scope="proposal_summary",
    task_title="Client proposal summary",
    raw_text="Summarize the proposal for the client",
    user_message="Write the summary",
)

session_id = prepared["session_id"]
guidance_context = prepared["guidance_context"]
inference_trace = prepared["inference_trace"]

feedback = adapter.save_feedback(
    session_id,
    assistant_message="A vague, abstract summary...",
    user_feedback="Use the client's actual business domain and write from their perspective.",
)

accepted = adapter.accept_response(
    session_id,
    task_type="proposal_summary",
    accepted_output="A client-facing summary with domain-specific language.",
)
```

## Memory Model

Core dataclasses live in [`schemas.py`](./src/claude_pseudo_intelligence/schemas.py):

- `EpisodeRecord`
- `CorrectionRecord`
- `ConversationTurn`
- `PreferenceRule`
- `RuleContext`
- `LatentContext`
- `TrainingExample`

Important rule fields:

- `context_mode`
  - `local`, `mixed`, or `general`
- `expected_gain`
  - How useful the rule tends to be when applied correctly
- `confidence_score`
  - How trustworthy the rule currently is
- `latent_contexts`
  - Learned situations where the rule has evidence

Important turn/session metadata:

- `reaction_score`
- `guidance_applied`
- `inference_trace`
- `authoritative_tags`
- `exclude_from_preference_rules`

## Retrieval Behavior

Guidance retrieval is not a simple keyword match anymore.

The system evaluates:

- scope match
- keyword overlap
- semantic similarity
- latent-context responsibility
- novelty probability
- posterior entropy
- expected gain
- confidence

If uncertainty is too high, the system can abstain instead of injecting noisy guidance.

## Growth Measurement

You can track whether the memory layer is actually improving outcomes.

```python
record = svc.record_growth(
    case_id="proposal-summary-quality",
    case_title="Proposal summary quality",
    task_scope="proposal_summary",
    baseline_output="generic summary",
    baseline_score=0.3,
    guided_output="client-facing, domain-specific summary",
    guided_score=0.85,
    guidance_text=guidance,
)
```

The server also supports automatic growth recording from conversation turns when both baseline and guided examples exist.

## Training Pipeline

Export accepted outputs as MLX-LM training data:

```bash
python3 scripts/export_training_dataset.py \
  --memory-dir /ABSOLUTE/PATH/TO/.pseudo-intelligence \
  --output-dir ./training_data
```

Outputs:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `preference.jsonl`
- `manifest.json`

Run automatic LoRA training:

```bash
pip install pseudo-intelligence-core[train]
python3 scripts/auto_train.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --memory-dir /ABSOLUTE/PATH/TO/.pseudo-intelligence \
  --output-dir ./training_artifacts
```

The auto-training flow tracks which episode IDs were already trained and can skip runs when too few new examples exist.

## Configuration

### Claude Code

```bash
claude mcp add pseudo-intelligence \
  -s user \
  -- python3 -m claude_pseudo_intelligence \
  --memory-dir ~/.pseudo-intelligence \
  --transport stdio
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pseudo-intelligence": {
      "command": "python3",
      "args": [
        "-m", "claude_pseudo_intelligence",
        "--memory-dir", "/ABSOLUTE/PATH/TO/.pseudo-intelligence",
        "--transport", "stdio"
      ]
    }
  }
}
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers."pseudo-intelligence"]
command = "/opt/homebrew/bin/python3"
args = [
  "/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/scripts/run_mcp_server.py",
  "--memory-dir",
  "/Users/setohirokazu/.pseudo-intelligence",
  "--transport",
  "stdio",
]
```

Example file:

- [`examples/codex_config.toml`](./examples/codex_config.toml)

### HTTP Transport

```bash
pseudo-intelligence \
  --memory-dir ~/.pseudo-intelligence \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --path /mcp
```

## Storage

All state is stored as JSON under the memory directory:

- `history.json`
- `conversation_history.json`
- `preference_rules.json`
- `chat_sessions/*.json`
- `growth/*.json`
- training output directories when export or auto-train is used

The store uses atomic file writes and backup recovery for the JSON files.

## Architecture

```text
src/claude_pseudo_intelligence/
  service.py              # Main facade
  mcp_server.py           # FastMCP server
  chat_adapter.py         # Session-oriented chat loop adapter
  history_store.py        # Persistent JSON storage
  schemas.py              # Dataclasses
  learning_context.py     # Retrieval and guidance construction
  conversation_learning.py # Lightweight correction extraction
  memory_manager.py       # Rule reconsolidation and latent context logic
  reaction_scorer.py      # Rule-based feedback scoring
  llm_scorer.py           # Optional LLM-backed reaction scoring
  growth_tracker.py       # Before/after measurement
  training_dataset.py     # MLX-LM dataset export
  auto_train.py           # Auto-training orchestration
  secret_store.py         # Secure secret storage
```

## What It Is Good At

- proposal and tender review flows
- brand and tone correction memory
- recurring editorial or UX feedback loops
- internal review assistance
- team-specific prompt guidance
- turning accepted outputs into supervised training data

## What It Is Not

- a full autonomous agent runtime
- a vector database knowledge platform
- a UI product by itself
- a substitute for domain-specific playbooks

## License

MIT

---

[Japanese documentation](docs/README_ja.md)
