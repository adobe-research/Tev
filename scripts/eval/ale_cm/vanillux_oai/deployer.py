"""VanilluxOaiDeployer: the tev training harness (single `bash` tool, vanillux prompts) as an
ale_run agent, over an OpenAI-compatible `/v1/chat/completions` endpoint (vLLM / SGLang).

It mirrors the training env so the ALE eval harness matches the training harness:
  - system template, observation truncation and format-error template:
    training/open-instruct/open_instruct/environments/vanillux_prompts.yaml
    (the instance template below omits that file's "Useful command examples" block)
  - single `bash` tool, SUBMIT_MARKER, persistent-shell wrapper:
    training/open-instruct/open_instruct/environments/swerl_vanillux_sandbox.py

Contract: one `bash` tool per turn (THOUGHT then the call); the agent submits by
running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`; cwd/env persist across calls;
observations are head/tail-truncated and end with `(exit_code=N)`. Config knobs
`parallel_tool_calls` / `execute_all_tool_calls` (both off by default) relax the
one-call contract for models that natively emit parallel tool calls. Scoring is done
afterwards by the ALE task's own grader (run by ale_run); this deployer only
drives the agent, it does NOT run tests.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import VanilluxOaiConfig

logger = logging.getLogger(__name__)
_TRANSCRIPT_NAME = "transcript.jsonl"

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# --- single bash tool (same schema as swerl_vanillux_sandbox.py) ---
BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. "
            "Working directory and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

# --- prompts (from vanillux_prompts.yaml; see module docstring) ---
SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer.\n\n"
    "Your response must include a THOUGHT section before your action where you\n"
    "explain your reasoning. After the THOUGHT, you must call the `bash` tool\n"
    "with EXACTLY ONE bash command (multiple commands chained with `&&` or `||`\n"
    "count as a single action).\n\n"
    "Failure to follow these rules — calling no tool, calling a tool other than\n"
    "`bash`, or omitting the THOUGHT — will cause your response to be rejected.\n"
)

INSTANCE_TEMPLATE = """Please solve this task:

{{task}}

You can execute bash commands and edit files (with `sed`, `cat > file << 'EOF'`,
etc.) to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your
changes and any possible problems.

1. Analyze the codebase / environment by finding and reading relevant files.
2. If applicable, create a script to reproduce the issue or expected behaviour.
3. Implement the change(s) by editing the source code or environment state.
4. Verify your fix works by running your script (or relevant test) again.
5. Test edge cases to ensure your fix is robust.
6. Submit your changes and finish your work by issuing the following command:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine it with any other command. After this command, you cannot
   continue working on this task.

## Important Rules

1. Every response must contain exactly one tool call to `bash`.
2. Directory and environment-variable changes ARE persistent across calls
   in this harness — you can `cd` and `export` and subsequent commands will
   see the change. (This differs from upstream mini-swe-agent's subshell
   model; treat the shell as a long-running login shell.)
3. Long-running commands: wrap with `timeout`, e.g. `timeout 30 <command>`.
4. Interactive commands are not possible. Use `yes`/`no` piping or
   non-interactive flags as appropriate.
5. Output may be truncated. Use `head`, `tail`, `grep`, `sed -n 'A,Bp'`,
   etc. to filter large outputs.
"""

TOO_LONG_HINT = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a\n"
    "smaller number of lines selectively. If you're using grep or find and it\n"
    "produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you\n"
    "can redirect output to a file and then search in that file."
)

FORMAT_ERROR_TEMPLATE = (
    "Format error: {{error}}\n\n"
    "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
    "end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "via the `bash` tool, with no other content in the command."
)

# --- persistent shell wrapper (mirrors swerl_vanillux_sandbox.py) ---
_CWD_PATH = "/tmp/.vanillux_cwd"
_ENV_PATH = "/tmp/.vanillux_env"
_WRAP_PATH = "/tmp/.vanillux_wrapper.sh"
_WRAPPER = f"""#!/bin/bash
set -a
source {_ENV_PATH} 2>/dev/null || true
set +a
_cwd="$(cat {_CWD_PATH} 2>/dev/null || echo /)"
cd "$_cwd" 2>/dev/null || cd / || exit 1
eval "$1"
_exit_code=$?
export -p > {_ENV_PATH}
pwd > {_CWD_PATH}
exit $_exit_code
"""


def render_instance(task: str) -> str:
    return INSTANCE_TEMPLATE.replace("{{task}}", task)


def format_error_message(error: str) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


def truncate_observation(output: str, cfg: VanilluxOaiConfig) -> str:
    if len(output) <= cfg.obs_max_chars:
        return output
    elided = len(output) - cfg.obs_head_chars - cfg.obs_tail_chars
    return (
        f"{TOO_LONG_HINT}\n\n"
        f"---- HEAD ({cfg.obs_head_chars} chars) ----\n"
        f"{output[:cfg.obs_head_chars]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({cfg.obs_tail_chars} chars) ----\n"
        f"{output[-cfg.obs_tail_chars:]}"
    )


class VanilluxOaiDeployer(BaseAgentDeployer):
    """The tev training harness (vanillux) over an OpenAI-compatible endpoint."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (_TRANSCRIPT_NAME,)

    @property
    def version(self) -> str | None:
        return "vanillux_oai-0.1.0"

    async def install(self) -> None:
        cfg: VanilluxOaiConfig = self.config  # type: ignore[assignment]
        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)
        base = cfg.server_url.rstrip("/").removesuffix("/v1")
        if cfg.health_check:
            try:
                with urllib.request.urlopen(base + "/health", timeout=30) as r:
                    r.read()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"vanillux_oai: server unreachable at {base}/health: {e}") from e
        logger.info("vanillux_oai install ok | server=%s | model=%s", cfg.server_url, cfg.model)

    def _call_model(self, messages: list[dict], cfg: VanilluxOaiConfig) -> dict:
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "tools": [BASH_TOOL],
            "tool_choice": "auto",
            "max_tokens": cfg.max_tokens,
        }
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            body["top_p"] = cfg.top_p
        if not cfg.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if cfg.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = cfg.parallel_tool_calls
        req = urllib.request.Request(
            cfg.server_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=cfg.request_timeout_s) as r:
            return json.loads(r.read())

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: VanilluxOaiConfig = self.config  # type: ignore[assignment]
        sb = self.executor.sandbox
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = work_dir / _TRANSCRIPT_NAME
        t0 = time.monotonic()

        def _flush(rec: dict) -> None:
            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Prepare the persistent shell in the sandbox: write the wrapper, seed cwd
        # from the sandbox's starting dir, start with an empty env snapshot.
        init_cmd = (
            f"cat > {_WRAP_PATH} << 'VANILLUX_WRAP_EOF'\n{_WRAPPER}\nVANILLUX_WRAP_EOF\n"
            f"chmod +x {_WRAP_PATH} && : > {_ENV_PATH} && pwd > {_CWD_PATH}"
        )
        try:
            await sb.run_command(init_cmd, timeout=cfg.command_timeout_s)
        except Exception as e:  # noqa: BLE001
            return AgentRunResult(status="failed", transcript_path=str(transcript_path),
                                  duration_s=round(time.monotonic() - t0, 2),
                                  error=f"persistent-shell init failed: {type(e).__name__}: {e}")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": render_instance(prompt)},
        ]
        _flush({"type": "instruction", "text": prompt})
        status, error, format_errors = "completed", None, 0

        try:
            for turn in range(cfg.max_turns):
                resp = await asyncio.to_thread(self._call_model, messages, cfg)
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                usage = resp.get("usage") or {}
                _flush({"type": "assistant", "turn": turn, "text": content, "tool_calls": tool_calls,
                        "finish_reason": choice.get("finish_reason"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens")})
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                # (tool_call, command-or-None) per call. Default vanillux contract keeps only
                # the first valid `bash` call; execute_all_tool_calls keeps every call.
                calls: list[tuple[dict, str | None]] = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    command = None
                    if fn.get("name") == "bash":
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        command = args.get("command") or None
                    if cfg.execute_all_tool_calls:
                        calls.append((tc, command))
                    elif command:
                        calls.append((tc, command))
                        break

                if not any(command for _, command in calls):
                    # format-error recovery (mirror training's TOOL_CALL_FORMAT_ERROR feedback)
                    format_errors += 1
                    if format_errors > cfg.max_format_errors:
                        error = f"exceeded max_format_errors ({cfg.max_format_errors})"
                        _flush({"type": "note", "text": error})
                        break
                    reason = ("Your last response did not include a valid `bash` tool call."
                              if not tool_calls else "You must call ONLY the `bash` tool.")
                    fmsg = format_error_message(reason)
                    messages.append({"role": "user", "content": fmsg})
                    _flush({"type": "format_error", "turn": turn, "text": fmsg})
                    continue

                submitted = False
                for i, (tc, command) in enumerate(calls):
                    call_id = tc.get("id") or f"call_{turn}_{i}"
                    if command is None:
                        obs = format_error_message("You must call ONLY the `bash` tool with a `command`.")
                        rc = None
                    else:
                        cp = await sb.run_command(f"bash {_WRAP_PATH} {shlex.quote(command)}",
                                                  timeout=cfg.command_timeout_s)
                        rc = cp.returncode
                        out = (cp.stdout or "") + (("\n" + cp.stderr) if cp.stderr else "")
                        if SUBMIT_MARKER in out:
                            _flush({"type": "submit", "turn": turn, "command": command})
                            submitted = True
                            break
                        obs = (truncate_observation(out, cfg) if out else "(no output)") + f"\n\n(exit_code={rc})"
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    _flush({"type": "observation", "turn": turn, "command": command,
                            "returncode": rc, "content": obs})
                if submitted:
                    break
            else:
                error = f"max_turns ({cfg.max_turns}) reached without submitting"
                _flush({"type": "note", "text": error})
        except Exception as e:  # noqa: BLE001
            import traceback
            status, error = "failed", f"{type(e).__name__}: {e}"
            _flush({"type": "error", "text": error, "traceback": traceback.format_exc()})
            logger.exception("vanillux_oai launch failed")

        return AgentRunResult(status=status, transcript_path=str(transcript_path),
                              duration_s=round(time.monotonic() - t0, 2), error=error)

    @classmethod
    def parse_artifacts(cls, *, work_dir: Path, config: Any, run_result: AgentRunResult,
                        builder: TrajectoryBuilder) -> None:
        transcript = work_dir / _TRANSCRIPT_NAME
        if not transcript.is_file():
            builder.add_step(source="system", message=f"vanillux_oai: no transcript at {transcript}",
                             extra={"reason": "parse_error"})
            return
        pending: list[str] = []
        for line in transcript.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rt = rec.get("type")
            if rt == "assistant":
                tcs, pending = [], []
                for tc in rec.get("tool_calls", []):
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    t = ToolCall(name=fn.get("name", "bash"), arguments=args)
                    tcs.append(t)
                    pending.append(t.id)
                builder.add_step(source="agent", message=rec.get("text") or None, tool_calls=tcs,
                                 metrics=StepMetrics(input_tokens=rec.get("prompt_tokens"),
                                                     output_tokens=rec.get("completion_tokens")))
            elif rt == "observation":
                cid = pending.pop(0) if pending else f"call_{rec.get('turn')}"
                builder.add_step(source="environment", observation=Observation(results=[ToolResult(
                    tool_call_id=cid, content=[ContentPart(type="text", text=rec.get("content", ""))],
                    is_error=bool(rec.get("returncode")))]))
            elif rt in ("submit", "format_error", "error", "note"):
                builder.add_step(source="system", message=rec.get("text") or rec.get("command"),
                                 extra={"type": rt})
