"""VanilluxOaiConfig: config for the vanillux training harness over an OpenAI-compatible /v1
endpoint.

Defaults reproduce the `swerl_vanillux_sandbox` RL-training rollout contract, so the eval
harness matches the training harness:
  * single `bash` tool, THOUGHT-then-bash system prompt
  * COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT submit marker
  * persistent shell (cd/export survive across calls)
  * head/tail observation truncation, format-error recovery
  * enable_thinking=True (the model was RL-trained with the interleaved-reasoning
    template, so it emits <think> then the bash tool call)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class VanilluxOaiConfig:
    name: ClassVar[str] = "vanillux_oai"

    # served-model-name + OpenAI-compatible base URL (vLLM/SGLang /v1).
    model: str = "qwen35-9b"
    server_url: str = "http://127.0.0.1:8008/v1"

    # Bearer token for the OpenAI-compatible endpoint. "sk-local" for an
    # unauthenticated local vLLM/SGLang; set to a real key (e.g. an
    # ${env:...} reference) to hit a hosted proxy/gateway (LiteLLM, etc.).
    api_key: str = "sk-local"
    # Preflight GET {server}/health before the run. True = fail-fast for a local
    # server that exposes /health; set False for a hosted gateway whose /health
    # needs admin auth or is absent (the first real call surfaces any error).
    health_check: bool = True

    # training: --max_steps 64 (bash calls per rollout).
    max_turns: int = 64
    # training rollout temperature was 1.0; per_turn_max_tokens 16384.
    temperature: float | None = 1.0
    top_p: float | None = None
    max_tokens: int = 16384
    # trained with interleaved reasoning -> keep thinking on for a faithful eval.
    enable_thinking: bool = True

    # give up after this many consecutive format errors (cf. Vanillux2Agent max_format_errors).
    max_format_errors: int = 64

    # Sent as the request's `parallel_tool_calls` when not None (None = omit, server default).
    parallel_tool_calls: bool | None = None
    # False = training contract: run only the first valid `bash` call of a turn.
    # True = run every tool call of a turn in order, answering each with its own tool
    # message (for models whose native format is parallel calls, e.g. GLM-5.3).
    execute_all_tool_calls: bool = False

    command_timeout_s: float = 900.0
    request_timeout_s: float = 1800.0

    # observation truncation (vanillux_prompts.yaml: max 10000, head/tail 5000).
    obs_max_chars: int = 10000
    obs_head_chars: int = 5000
    obs_tail_chars: int = 5000
