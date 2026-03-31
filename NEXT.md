# Next Step

## Current state

- `claude_pseudo_intelligence_core` now stores:
  - `EpisodeRecord`
  - `CorrectionRecord`
  - `ConversationTurn`
  - `PreferenceRule`
- `build_guidance_context(...)` already merges:
  - case-based correction memory
  - conversation-based preference memory
- MCP server already exists
- accepted outputs can already export:
  - `train.jsonl`
  - `valid.jsonl`
  - `test.jsonl`
  - `preference.jsonl`

## Core idea

Do not store whole conversations as truth.

Use:

- `conversation -> extracted correction -> promoted rule -> reused guidance`

## Next implementation

Build an adapter layer that automatically does three things during chat:

1. call `build_guidance_context(...)` before response generation
2. call `save_conversation_turn(...)` after the user gives corrective feedback
3. call `save_training_example(...)` when a final accepted answer is confirmed

## Preferred shape

- first choice: MCP server
- second choice: thin CLI adapter

## Goal

Make the system grow from repeated dialogue without turning into a raw chat log warehouse.
