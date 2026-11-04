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
     stock. Whether the pinned-host backend runs at all is upstream's
     decision (``engram_config.cpu_offload`` selects it at the construction
     site); the plugin only swaps which class implements it.
  2. ``Qwen4ExpNGramEmbedding.load_weights`` is replaced with a copy that
     routes the int4 table's ``shard_{i}.weight_scale`` tensors through the
     PLE shard loader and refuses unsharded or miscast packed tables.

The int4 lookup itself rides upstream's ``lookup_dtype`` /
``lookup_from_pinned`` seam on the embedding method: no ``_lookup`` override
and no eager prefetch buffer. KV-offload group selection is left to the
in-tree offloading config; the plugin installs no group filter.

A source-hash assert pins the exact vLLM tree this was authored against;
wheel drift fails loudly at launch instead of subtly at weight load.
"""

from __future__ import annotations

import hashlib
import inspect
import threading

_LOCK = threading.Lock()
_INSTALLED = False

# SHA-256 of the installed vllm/models/qwen4_exp/nvidia/ngram_embedding.py
# that this plugin was authored against (v0.33.0 base). The pinned nvidia
# file still constructs Qwen4ExpPLEPinnedHostEmbedding through its own
# module globals (lines 209 to 238, from the from_quant_config call through
# the embedding_cls(...) construction), so the global rebind, classmethod
# patch and load_weights replacement keep working.
PINNED_NGRAM_SHA256 = "8442e161d90d09ccc0aab06aea809e8f1fc387c8af348a32a9424bfa1f755533"

# SHA-256 of the vllm/models/qwen4_exp/common/ngram_embedding.py module
# that holds the PLE base classes and the lookup seam the plugin rides.
# Drift here must refuse install just like drift in nvidia/ does.
PINNED_COMMON_NGRAM_SHA256 = (
    "ef0319508554d26da0870c5b8894d4a96b6d82ad069695558d2497c6814102f8"
)

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
        from ple_int4.method import patched_load_weights

        actual = _sha256_of_source(ng)
        if PINNED_NGRAM_SHA256 != "TODO" and actual != PINNED_NGRAM_SHA256:
            raise RuntimeError(
                "ple_int4: vLLM's ngram_embedding.py does not match the pinned "
                f"nightly (got sha256 {actual[:16]}..., pinned "
                f"{PINNED_NGRAM_SHA256[:16]}...). Re-author the patches against "
                "the installed wheel or pin the matching nightly."
            )

        import vllm.models.qwen4_exp.common.ngram_embedding as common_ng

        actual_common = _sha256_of_source(common_ng)
        if (
            PINNED_COMMON_NGRAM_SHA256 != "TODO"
            and actual_common != PINNED_COMMON_NGRAM_SHA256
        ):
            raise RuntimeError(
                "ple_int4: vLLM's common/ngram_embedding.py does not match the "
                f"pinned nightly (got sha256 {actual_common[:16]}..., pinned "
                f"{PINNED_COMMON_NGRAM_SHA256[:16]}...). Re-author the patches "
                "against the installed wheel or pin the matching nightly."
            )

        from ple_int4.method import (
            Qwen4ExpPLEInt4EmbeddingMethod,
            Qwen4ExpPLEPinnedHostInt4Embedding,
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


# The TP4 all-reduce force lives in this fork's source tree: the platform
# layer accepts PCIe P2P as fully connected and the communicator allows the
# EP group through. The plugin does not patch it, and the
# expandable_segments exclusion stays documented alongside the
# custom-all-reduce incompatibility.


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
