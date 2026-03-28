#!/usr/bin/env python3
"""PreToolUse hook: block destructive process cleanup unless explicitly confirmed.

Confirmation token required in tool arguments/command:
  CONFIRM_DESTRUCTIVE_CLEANUP=YES
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

CONFIRM_TOKEN = "CONFIRM_DESTRUCTIVE_CLEANUP=YES"

# Focused on process cleanup operations that can kill unrelated jobs.
DESTRUCTIVE_PATTERNS = [
    r"\bkillall\b",
    r"\bpkill\b",
    r"\bkill\s+-9\b",
    r"\bkill\s+--signal\s+KILL\b",
    r"\bxargs\s+kill\b",
    r"\bpkill\s+-f\b",
    r"\bkill\s+\$\(cat\s+[^)]*\.pid\)",
    r"\brm\s+-f\s+[^\n]*\.pid\b",
    r"\brm\s+-rf\s+logs\b",
]


def _collect_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_collect_strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_collect_strings(v))
    return out


def _extract_tool_name(payload: dict[str, Any]) -> str:
    for key in ("toolName", "tool_name", "name"):
        val = payload.get(key)
        if isinstance(val, str):
            return val
    tool_input = payload.get("toolInput")
    if isinstance(tool_input, dict):
        val = tool_input.get("toolName") or tool_input.get("name")
        if isinstance(val, str):
            return val
    return ""


def _is_destructive(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in DESTRUCTIVE_PATTERNS)


def _decision_allow() -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": "No destructive cleanup pattern detected or explicit confirmation token present.",
        }
    }


def _decision_deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "systemMessage": (
            "Destructive process cleanup blocked. Add explicit confirmation token "
            f"'{CONFIRM_TOKEN}' to the command arguments, then retry."
        ),
    }


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps(_decision_allow()))
        return 0

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Fail-open to avoid deadlocking sessions on malformed hook input.
        print(json.dumps(_decision_allow()))
        return 0

    all_text = "\n".join(_collect_strings(payload))
    if CONFIRM_TOKEN in all_text:
        print(json.dumps(_decision_allow()))
        return 0

    tool_name = _extract_tool_name(payload).lower()
    # Apply strict checks to command-execution tools.
    if tool_name and not any(x in tool_name for x in ("terminal", "execute", "task", "shell", "run_in_terminal")):
        print(json.dumps(_decision_allow()))
        return 0

    if _is_destructive(all_text):
        reason = (
            "Detected destructive cleanup command pattern (killall/pkill/kill -9/rm pid/logs). "
            f"Explicit confirmation required: {CONFIRM_TOKEN}."
        )
        print(json.dumps(_decision_deny(reason)))
        return 0

    print(json.dumps(_decision_allow()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
