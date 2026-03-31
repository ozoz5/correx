# Latest Research Notes (2026-03-28)

## What mattered

The important shift in 2025-2026 is not "more memory".

It is:

- scoped memory
- procedural guidance
- correction-to-rule promotion
- MCP-native transport
- memory-first, fine-tune-later training operations

## What was adopted

### 1. Procedural guidance is now first-class

`PreferenceRule` is no longer just free text.

It now carries:

- `instruction`
- `applies_to_scope`
- `applies_when_tags`
- `negative_conditions`
- `priority`
- `version`

This follows the direction seen in LangMem-style procedural memory.

### 2. Correction records are richer

`CorrectionRecord` now supports:

- `reason`
- `scope`
- `bad_output`
- `revised_output`
- `tool_used`
- `source_user`
- `accepted`

This keeps future promotion / rollback / evaluation possible.

### 3. MCP primitives are less mixed

Promoted guidance is now exposed as a read-only resource:

- `memory://guidance/{task_scope}`

This follows the MCP direction of separating:

- resources for application-driven context
- tools for model-driven actions

### 4. Training data is future-compatible

`TrainingExample` now stores:

- `rejected_output`
- `model_id`
- `policy_version`
- `accepted_by`
- `tags`
- `temperature`

Dataset export now emits:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `preference.jsonl`

Default dataset split is now `chronological`.

## Why this matters

This core is no longer only:

- memory
- correction notes

It is moving toward:

- correction events
- procedural guidance
- reusable supervision artifacts

## Source anchors

- Anthropic Claude Code memory:
  [https://code.claude.com/docs/en/memory](https://code.claude.com/docs/en/memory)
- Anthropic Claude Code subagents:
  [https://code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)
- Anthropic Claude Code hooks:
  [https://code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)
- Anthropic memory tool:
  [https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- Anthropic prompt caching:
  [https://platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- MCP tools spec:
  [https://modelcontextprotocol.io/specification/2025-11-25/server/tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- MCP resources spec:
  [https://modelcontextprotocol.io/specification/2025-11-25/server/resources](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)
- MCP prompts spec:
  [https://modelcontextprotocol.io/specification/2025-11-25/server/prompts](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts)
- MCP authorization:
  [https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- LangMem conceptual guide:
  [https://langchain-ai.github.io/langmem/concepts/conceptual_guide/](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- LangMem prompt optimization:
  [https://langchain-ai.github.io/langmem/guides/optimize_memory_prompt/](https://langchain-ai.github.io/langmem/guides/optimize_memory_prompt/)
- Mem0 MCP integration:
  [https://docs.mem0.ai/platform/features/mcp-integration](https://docs.mem0.ai/platform/features/mcp-integration)
- MLX:
  [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- MLX-LM:
  [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- MLX-LM LoRA:
  [https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- TRL dataset formats:
  [https://huggingface.co/docs/trl/dataset_formats](https://huggingface.co/docs/trl/dataset_formats)
