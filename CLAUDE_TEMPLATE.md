# CORREX Auto-Learning Loop (paste this into your CLAUDE.md)

## Automatic Correction Memory

When the correx MCP server is connected, follow these rules in every session.

### 1. Session Start — Load Past Learnings

Before starting non-trivial tasks, call `build_guidance_context`:
```
build_guidance_context(task_title="...", raw_text="...")
```
Follow the returned rules in your generation.

Also call `get_cognitive_map` to check knowledge gap areas.
In scopes with high escalation, explain things more thoroughly.

### 2. When the User Corrects You — Record Immediately

If the user gives feedback (corrections, praise, direction changes),
call `save_conversation_turn` in that same turn:

```
save_conversation_turn(
  task_scope="task type",
  user_message="what the user said",
  assistant_message="what you said",
  user_feedback="the user's feedback",
  extracted_corrections=["generalized rule for future use"],
  guidance_applied=true/false,
  reaction_score_override=0.75
  # 0.0=strong rejection / 0.3=unhappy / 0.5=neutral / 0.75=approval / 0.9=strong praise
)
```

### 3. When You Detect Questions — Save Curiosity Signal

If the user's message contains a question, call `save_curiosity_signal`:
```
save_curiosity_signal(
  question_text="the question",
  question_type="knowledge_gap",  # or judgment_uncertainty / confirmation_seeking
  target="self",                  # or "other" for translation tasks
  task_scope="scope",
  keywords=["keyword1", "keyword2"],
  confidence=0.8
)
```

When the user is satisfied, resolve the clusters:
```
resolve_curiosity_clusters(task_scope="resolved scope")
```

### 4. Task Completion — Save Episode

When meaningful work is completed:
```
save_episode(title="what was done", task_type="type", output={...})
```

### 5. Self-Evaluation — Rate Rule Effectiveness

After completing a task with guidance, evaluate each rule:
```
evaluate_guidance_effectiveness(
  evaluations=[{"rule_id": "pref-xxx", "score": 0.9, "reason": "why"}],
  task_scope="scope"
)
```
