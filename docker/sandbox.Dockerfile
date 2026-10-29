# Agent sandbox image for Tev-Gym tasks (the image each dataset row names in env_config.image).
#
# It contains only what the AGENT may use (python 3.13, the agent-permitted libraries, gcc/binutils/sqlite3).
# It deliberately contains NO grader code: grading runs host-side in the dense env
# (swerl_vanillux_dense_sandbox, require_grader=True) from {task_data_dir}/{task_id}/authority/package, so
# nothing the agent can read in the container reveals a task's reference answer. Do not add the grader
# codebase (training_data_collection) to this image — an agent could import the reference solutions.
#
# The open-instruct DockerBackend execs the sandbox as the image's default user with no --user, and the harness
# does `mkdir -p /workspace /output /logs/verifier` at submit, so the default user must be root
# (python:*-slim already is).
FROM python:3.13-slim

# System tools required by the tasks' declared toolchains, e.g.:
#   gcc/cc     -> code.adversarial-program-tests (compiles C submissions via the harness)
#                 NB: --no-install-recommends skips libc6-dev, so cc can compile (-c) but not link
#                 (no stdio.h / crt1.o). Kept for parity with the image used in training.
#   binutils   -> code.elf-configuration-extraction (readelf)
#   sqlite3    -> cm.warehouse-etl
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        binutils \
        sqlite3 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned agent-permitted libraries. The two unpinned transitive deps are pinned too, so `pip freeze`
# matches the image used in training exactly.
COPY requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sandbox.txt Pygments==2.20.0 typing-inspection==0.4.3

# Fail the build if the deps are missing or any copy of the grader codebase is reachable.
RUN python3 -c "import numpy, safetensors, pandas, pydantic, yaml, cryptography, pytest; print('deps OK')" \
    && ! python3 -c "import training_data_collection" 2>/dev/null \
    && [ ! -e /opt/tdc ] \
    && [ -z "$(find / -xdev -name 'training_data_collection*' -print -quit 2>/dev/null)" ] \
    && echo 'sandbox image OK: no grader codebase'

WORKDIR /workspace
