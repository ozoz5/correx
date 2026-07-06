"""Rule status lifecycle — transitions recorded as events, not overwrites.

All status changes in the correx system go through transition_status().
No other module should assign rule.status / rule["status"] directly.

Design (adapted from bi-temporal fact edges): an outdated status is not
erased — the transition is appended to status_history so that
  - status churn (demote <-> promote oscillation) becomes observable,
  - concurrent writers are identifiable per event (writer_pid),
  - "what was this rule's status at time T" stays answerable.
"""
from __future__ import annotations

import os
from datetime import datetime

MAX_HISTORY = 20


def transition_status(rule, new_status: str, reason: str = ""):
    """Change a rule's status, appending an event to its status_history.

    Accepts both dict-shaped rules (raw JSON items) and PreferenceRule
    dataclass instances. No-op when the status is unchanged. Returns the
    same rule object for call-site convenience.
    """
    is_dict = isinstance(rule, dict)
    old = rule.get("status", "") if is_dict else getattr(rule, "status", "")
    if old == new_status:
        return rule

    event = {
        "at": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "from": old,
        "to": new_status,
        "reason": reason[:120],
        "writer_pid": os.getpid(),
    }

    if is_dict:
        history = rule.setdefault("status_history", [])
        if not isinstance(history, list):
            history = []
            rule["status_history"] = history
        history.append(event)
        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]
        rule["status"] = new_status
    else:
        history = getattr(rule, "status_history", None)
        if isinstance(history, list):
            history.append(event)
            if len(history) > MAX_HISTORY:
                del history[:-MAX_HISTORY]
        rule.status = new_status
    return rule
