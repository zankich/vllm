# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for the CPU offload tier tests."""

import pytest


@pytest.fixture(autouse=True)
def _set_spawn_method(monkeypatch):
    # Keep the multiprocessing start method explicit across the CPU-tier
    # suites: fork-interested extensions (NVML handles on WSL, CUDA
    # contexts) misbehave under an implicitly-forked worker, and the
    # shared-region tests spawn real worker processes.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
