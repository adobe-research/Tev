"""Vanillux prompt harness used by :mod:`Vanillux2Agent.agent`.

Vendored from the vanillux solver of TMax (https://github.com/hamishivi/tmax, arXiv:2606.23321) so the
TB2.1 eval agent is self-contained: the bash tool schema,
submit marker, tool-call parsing, and the mini-SWE-agent-derived prompts / observation truncation /
format-error recovery. Import-light on purpose (json/yaml only) so it loads inside the Harbor venv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command. Each command runs in a new subshell.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
            },
        },
    }
]


def _extract_tool_call(response_msg: dict) -> Dict[str, Optional[str]]:
    """Parse a tool-calling response message.

    Returns dict with:
      type: "command" | "done" | "no_tool_call"
      command: the bash command string (if type=="command")
      tool_call_id: the id needed for the tool response message
    """
    tool_calls = response_msg.get("tool_calls")
    if not tool_calls:
        return {"type": "no_tool_call", "command": None, "tool_call_id": None}

    tc = tool_calls[0]
    func = tc.get("function", {})
    func_name = func.get("name", "")

    if func_name != "bash":
        return {"type": "no_tool_call", "command": None, "tool_call_id": tc.get("id")}

    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return {"type": "no_tool_call", "command": None, "tool_call_id": None}
    else:
        args = args_raw

    command = args.get("command", "").strip()
    tool_call_id = tc.get("id")

    if SUBMIT_MARKER in command:
        return {"type": "done", "command": command, "tool_call_id": tool_call_id}

    return {"type": "command", "command": command, "tool_call_id": tool_call_id}


_PROMPTS_PATH = Path(__file__).resolve().parent / "vanillux_prompts.yaml"


def _load_vanillux_prompts() -> Dict[str, Any]:
    """Load the vendored vanillux prompts. Cached at module level."""
    with _PROMPTS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_VANILLUX_PROMPTS: Dict[str, Any] = _load_vanillux_prompts()
_SYSTEM_TEMPLATE: str = _VANILLUX_PROMPTS["system_template"]
_INSTANCE_TEMPLATE: str = _VANILLUX_PROMPTS["instance_template"]
_OBS_CFG: Dict[str, Any] = _VANILLUX_PROMPTS["observation"]
_FORMAT_ERROR_TEMPLATE: str = _VANILLUX_PROMPTS["format_error_template"]


def _render_instance(task: str) -> str:
    """Render the vanillux instance template with the task description.

    The vendored template uses ``{{task}}`` as the only Jinja variable; we
    substitute literally rather than pulling in Jinja2 to keep deps light.
    """
    return _INSTANCE_TEMPLATE.replace("{{task}}", task)


def _truncate_observation(output: str) -> str:
    """Apply mini-swe-agent's head/tail truncation to long tool outputs."""
    max_chars = int(_OBS_CFG.get("max_chars", 10000))
    if len(output) <= max_chars:
        return output
    head_n = int(_OBS_CFG.get("head_chars", 5000))
    tail_n = int(_OBS_CFG.get("tail_chars", 5000))
    elided = len(output) - head_n - tail_n
    hint = _OBS_CFG.get("too_long_hint", "Output truncated.")
    return (
        f"{hint}\n\n"
        f"---- HEAD ({head_n} chars) ----\n"
        f"{output[:head_n]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({tail_n} chars) ----\n"
        f"{output[-tail_n:]}"
    )


def _format_error_message(error: str) -> str:
    return _FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)
