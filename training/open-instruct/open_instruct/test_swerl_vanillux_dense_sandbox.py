"""Tests for the dense sandbox's per-turn PPRS shaping λ·(γΦₜ−Φₜ₋₁) (λ = BETA below) and host-side terminal reward."""

import os
import tempfile
import unittest

from open_instruct.environments.backends import ExecutionResult
from open_instruct.environments.base import EnvCall
from open_instruct.environments.swerl_vanillux_dense_sandbox import SWERLVanilluxDenseSandboxEnv
from open_instruct.environments.swerl_vanillux_sandbox import SUBMIT_MARKER

BETA = 0.25


class _WorkspaceBackend:
    """Fake container: `set <phi>` moves the workspace to a state of potential phi; submit echoes the marker."""

    def __init__(self):
        self.phi: float | None = None  # None = /app/output not created yet

    def run_command(self, command: str) -> ExecutionResult:
        if "set " in command:
            self.phi = float(command.split("set ", 1)[1].split("'", 1)[0])
        if "SUBMIT" in command:
            return ExecutionResult(stdout=SUBMIT_MARKER + "\n", stderr="", exit_code=0)
        return ExecutionResult(stdout="ok", stderr="", exit_code=0)


class _FakeGrader:
    """Φ and the terminal score are read back from the extracted output tree."""

    def __init__(self, terminal: float):
        self.terminal_score = terminal
        self.progress_calls = 0

    def progress(self, output_dir: str) -> dict:
        self.progress_calls += 1
        with open(os.path.join(output_dir, "state.txt")) as f:
            return {"potential": float(f.read())}

    def terminal_full(self, output_dir: str) -> dict:
        return {"total": self.terminal_score, "component_values": None, "component_pass_mean": None}

    def close(self) -> None:
        pass


def _make_env(terminal: float = 0.6, grade_mode: str = "live") -> SWERLVanilluxDenseSandboxEnv:
    env = SWERLVanilluxDenseSandboxEnv()
    backend = _WorkspaceBackend()
    env._backend = backend
    env._grader = _FakeGrader(terminal)
    env._beta = BETA
    env._active_grade_mode = grade_mode

    def _extract_output(container_path: str = "/app/output") -> str | None:
        if backend.phi is None:
            return None
        out = tempfile.mkdtemp(prefix="v72-test-out-")
        with open(os.path.join(out, "state.txt"), "w") as f:
            f.write(repr(backend.phi))
        return out

    env._extract_output = _extract_output
    return env


async def _run(env: SWERLVanilluxDenseSandboxEnv, commands: list[str]) -> list[float]:
    rewards = []
    for i, command in enumerate(commands):
        result = await env.step(EnvCall(id=str(i), name="bash", args={"command": command}))
        rewards.append(result.reward)
        if result.done:
            break
    return rewards


# States 0.2, 0.2 (unchanged), 0.5, 0.4 (regression), 0.7, then submit with terminal R = 0.6.
TRAJECTORY = ["echo set 0.2", "echo set 0.2", "echo set 0.5", "echo set 0.4", "echo set 0.7"]


class TestPerTurnShaping(unittest.IsolatedAsyncioTestCase):
    async def test_per_turn_rewards_telescope_to_final_potential(self):
        env = _make_env()
        rewards = await _run(env, TRAJECTORY + ["echo SUBMIT"])
        self.assertEqual(len(rewards), 6)
        self.assertAlmostEqual(rewards[0], BETA * 0.2)
        self.assertAlmostEqual(rewards[1], 0.0)
        self.assertAlmostEqual(rewards[3], BETA * -0.1)
        self.assertAlmostEqual(rewards[-1], 0.6)
        self.assertAlmostEqual(sum(rewards), 0.6 + BETA * 0.7)
        # 4 distinct states graded per turn (the repeated 0.2 is a hash hit) + the submit-time grade.
        self.assertEqual(env._grader.progress_calls, 5)
        metrics = env.get_metrics()
        self.assertAlmostEqual(metrics["progress_potential_final"], 0.7)
        self.assertAlmostEqual(metrics["terminal_reward_raw"], 0.6)
        self.assertEqual(metrics["progress_grade_calls"], 5.0)

    async def test_non_submitting_rollout_banks_per_turn_deltas(self):
        env = _make_env()
        rewards = await _run(env, TRAJECTORY)
        self.assertEqual(len(rewards), 5)
        self.assertAlmostEqual(sum(rewards), BETA * 0.7)
        self.assertEqual(env._grader.progress_calls, 4)

    async def test_terminal_only_tasks_get_terminal_reward_only(self):
        env = _make_env(grade_mode="terminal-only")
        rewards = await _run(env, TRAJECTORY + ["echo SUBMIT"])
        self.assertEqual(rewards[:-1], [0.0] * 5)
        self.assertAlmostEqual(rewards[-1], 0.6)
        self.assertEqual(env._grader.progress_calls, 1)  # only the submit-time grade (logged, not paid)


if __name__ == "__main__":
    unittest.main()
