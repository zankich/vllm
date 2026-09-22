# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ple_int4: INT4 PLE embedding for Qwen3.8-Flash-Next on vLLM.

Loaded in every vLLM process (engine core + TP workers) via the
``vllm.general_plugins`` entry point. install() makes two patches to the
in-memory ``vllm.models.qwen4_exp.nvidia.ngram_embedding`` module — no files
in the vLLM installation are modified:

  1. ``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` is wrapped to return the
     int4 method when the model config carries ``ple_embedding_dtype == "int4"``
     (the marker our packer writes); anything else delegates to stock. The
     same wrapper rebinds the module global ``Qwen4ExpPLEPinnedHostEmbedding``
     to the int4-aware subclass before the construction site resolves it;
     for non-int4 checkpoints the subclass is behaviorally identical to
     stock.
  2. ``Qwen4ExpNGramEmbedding.load_weights`` is replaced with a copy that
     derives shard shape checks from the destination tensors and routes
     ``shard_{i}.weight_scale`` tensors through the PLE shard loader.

KV-offload group selection is left to the in-tree offloading config; the
plugin installs no group filter.

A source-hash assert pins the exact vLLM nightly this was authored against;
wheel drift fails loudly at launch instead of subtly at weight load.
"""

from __future__ import annotations

import hashlib
import inspect
import threading

_LOCK = threading.Lock()
_INSTALLED = False

# SHA-256 of the installed vllm/models/qwen4_exp/nvidia/ngram_embedding.py
# that this plugin was authored against (nightly 0.29.1rc1.dev102+gba2ae9f23;
# verified byte-identical to checkout c69d5d72a6 on 2026-09-15).
# nightly wheel pin was f3aaf292... ; v0.29.0z pin:
PINNED_NGRAM_SHA256 = "f3aaf29281b803dcebb4b179d5177704ed7b913170edf8107e33f729be825429"

_ORIGINALS: dict[str, object] = {}


def _sha256_of_source(module) -> str:
    src = inspect.getsource(module)
    return hashlib.sha256(src.encode()).hexdigest()


def install() -> None:
    """Apply the in-memory patches (idempotent, per-process)."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng
        from ple_int4.method import (
            Qwen4ExpPLEInt4EmbeddingMethod,
            Qwen4ExpPLEPinnedHostInt4Embedding,
            patched_load_weights,
        )

        actual = _sha256_of_source(ng)
        if PINNED_NGRAM_SHA256 != "TODO" and actual != PINNED_NGRAM_SHA256:
            raise RuntimeError(
                "ple_int4: vLLM's ngram_embedding.py does not match the pinned "
                f"nightly (got sha256 {actual[:16]}..., pinned "
                f"{PINNED_NGRAM_SHA256[:16]}...). Re-author the patches against "
                "the installed wheel or pin the matching nightly."
            )

        original_fqc = ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config
        stock_pinned_host = ng.Qwen4ExpPLEPinnedHostEmbedding

        def _from_quant_config(quant_config, prefix, embedding_dtype=None):
            if embedding_dtype == "int4":
                # Bind the int4 pinned-host backend before the construction
                # site resolves the module global.
                ng.Qwen4ExpPLEPinnedHostEmbedding = Qwen4ExpPLEPinnedHostInt4Embedding
                return Qwen4ExpPLEInt4EmbeddingMethod()
            # Non-int4 config: restore stock so this plugin changes nothing.
            ng.Qwen4ExpPLEPinnedHostEmbedding = stock_pinned_host
            return original_fqc(quant_config, prefix, embedding_dtype)

        _ORIGINALS["from_quant_config"] = original_fqc
        _ORIGINALS["pinned_host_embedding"] = stock_pinned_host
        _ORIGINALS["load_weights"] = ng.Qwen4ExpNGramEmbedding.load_weights

        ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(
            _from_quant_config
        )
        ng.Qwen4ExpNGramEmbedding.load_weights = patched_load_weights

        _INSTALLED = True


# The TP4 all-reduce force moved into the fork source (2026-09-20):
# NvmlCudaPlatform.is_fully_connected accepts P2P read/write when NVLink
# is absent, and cuda_communicator allows the ep group. The plugin no
# longer patches it — the probe in source asks the hardware instead of
# forcing True, and the expandable_segments exclusion stays documented
# in the composes (custom_all_reduce.cuh:164).


def uninstall() -> None:
    """Restore stock behavior (tests only)."""
    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            return
        import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng

        ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config = _ORIGINALS[
            "from_quant_config"
        ]
        ng.Qwen4ExpPLEPinnedHostEmbedding = _ORIGINALS["pinned_host_embedding"]
        ng.Qwen4ExpNGramEmbedding.load_weights = _ORIGINALS["load_weights"]
        _INSTALLED = False
