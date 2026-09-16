"""ple_int4: INT4 PLE embedding for Qwen3.8-Flash-Next on vLLM.

Loaded in every vLLM process (engine core + TP workers) via the
``vllm.general_plugins`` entry point. install() makes three patches to the
in-memory ``vllm.models.qwen4_exp.nvidia.ngram_embedding`` module — no files
in the vLLM installation are modified:

  1. ``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` is wrapped to return the
     int4 method when the model config carries ``ple_embedding_dtype == "int4"``
     (the marker our packer writes); anything else delegates to stock.
  2. The module global ``Qwen4ExpPLEPinnedHostEmbedding`` is rebound to our
     int4-aware subclass; the backend-selection site in
     ``Qwen4ExpNGramEmbedding.__init__`` resolves that global at call time.
     For non-int4 checkpoints the subclass is behaviorally identical to stock.
  3. ``Qwen4ExpNGramEmbedding.load_weights`` is replaced with a copy that
     derives shard shape checks from the destination tensors and routes
     ``shard_{i}.weight_scale`` tensors through the PLE shard loader.

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
# nightly wheel pin was f3aaf292... ; v0.29.0-qwen-flashnext e94e5cf6e pin:
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

        def _from_quant_config(quant_config, prefix, embedding_dtype=None):
            if embedding_dtype == "int4":
                return Qwen4ExpPLEInt4EmbeddingMethod()
            return original_fqc(quant_config, prefix, embedding_dtype)

        _ORIGINALS["from_quant_config"] = original_fqc
        _ORIGINALS["pinned_host_embedding"] = ng.Qwen4ExpPLEPinnedHostEmbedding
        _ORIGINALS["load_weights"] = ng.Qwen4ExpNGramEmbedding.load_weights

        ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(
            _from_quant_config
        )
        ng.Qwen4ExpPLEPinnedHostEmbedding = Qwen4ExpPLEPinnedHostInt4Embedding
        ng.Qwen4ExpNGramEmbedding.load_weights = patched_load_weights

        _install_tp4_allreduce_patch()
        _install_offload_hybrid_patch()
        _INSTALLED = True


def _install_offload_hybrid_patch() -> None:
    """Exclude block-misaligned KV groups from offload selection.

    The upstreamed OffloadingConnector asserts every selected group's
    tokens_per_block divides the cross-group hash LCM. On Flash-Next + MTP
    the groups are [800, 800, 800, 800, 8, 800]: the drafter attention's
    8-token block trips the assert at init (observed 2026-09-16,
    offloading/config.py build_offloading_config). This patch drops groups
    whose block size does not divide the largest selected block size —
    exactly the drafter group here — before selection, so the main model's
    context still offloads. the v0.29.0-qwen branch solves the same problem with
    drafter-group annotation; upstream has no knob yet.

    Harmless when KV offload is off (the function is never called).
    Disable with PLE_INT4_OFFLOAD_HYBRID=0.
    """
    import logging
    import os

    if os.environ.get("PLE_INT4_OFFLOAD_HYBRID", "1") == "0":
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            config as _oc,
        )
    except ImportError:  # pragma: no cover - older wheels
        return
    if not hasattr(_oc, "get_offloading_group_ids"):
        # The fork (v0.29.0-qwen) predates upstream's group-selection
        # refactor; no assert to dodge there. Installing nothing also keeps
        # plugin load from crashing on wheels where the module exists but
        # the symbol does not (observed 2026-09-16: AttributeError
        # during `vllm serve --help` via plugin load).
        return

    original = _oc.get_offloading_group_ids

    def _aligned_group_ids(kv_cache_config):
        ids = list(original(kv_cache_config))
        groups = kv_cache_config.kv_cache_groups
        sizes = {
            gid: _oc.resolve_dcp_kv_block_size(groups[gid].kv_cache_spec, 1)
            for gid in ids
        }
        logging.getLogger(__name__).warning(
            "ple_int4 offload-filter diagnostic: ids=%s sizes=%s",
            ids, sizes,
        )
        if not sizes:
            return tuple(ids)
        largest = max(sizes.values())
        # The connector asserts tokens_per_block % tokens_per_hash == 0 for
        # every selected group: block size must be a MULTIPLE of the hash
        # LCM. Keep only those; the drafter's 8-token block fails 8 % 800.
        keep = [gid for gid in ids if sizes[gid] % largest == 0]
        dropped = [gid for gid in ids if gid not in keep]
        if dropped:
            logging.getLogger(__name__).warning(
                "ple_int4: excluding %d block-misaligned KV group(s) %s from "
                "offload selection (drafter attention); main groups offload normally",
                len(dropped),
                dropped,
            )
        return tuple(keep)

    _ORIGINALS["offload_group_ids"] = (_oc, original)
    _oc.get_offloading_group_ids = _aligned_group_ids


def _install_tp4_allreduce_patch() -> None:
    """Force CudaPlatform.is_fully_connected() -> True (TP4 without NVLink).

    vLLM gates the CUSTOM all-reduce path on `world_size > 2 and not
    fully_connected`, and that probe asks for NVLink specifically. Four
    PCIe-only 3090 Ti with working P2P across every pair (`nvidia-smi topo
    -p2p r/w` OK), so the gate rejects a path the hardware supports and TP4
    falls back to PYNCCL alone. Measured +4.5% single-stream decode at TP2
    on this box; the win is decode's small-message latency, not prefill.

    The in-image equivalent is the `sed` in the deploy repo's TP4 composes; doing it here keeps the zero-vLLM-tree-edit property
    and reaches every worker through the same plugin entry point.

    HARD CONSTRAINT: mutually exclusive with
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments — VMM ranges break
    cudaIpcGetMemHandle and workers die at custom_all_reduce.cuh:164
    'invalid argument'. Never set that env with this patch active.

    Disable with PLE_INT4_TP4_ALLREDUCE=0.
    """
    import os

    if os.environ.get("PLE_INT4_TP4_ALLREDUCE", "1") == "0":
        return
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").find("expandable_segments") >= 0:
        raise RuntimeError(
            "ple_int4: expandable_segments is mutually exclusive with the CUSTOM "
            "all-reduce path (custom_all_reduce.cuh:164). Unset "
            "PYTORCH_CUDA_ALLOC_CONF or set PLE_INT4_TP4_ALLREDUCE=0."
        )
    try:
        from vllm.platforms.cuda import CudaPlatform
    except ImportError:  # pragma: no cover
        return
    _ORIGINALS["is_fully_connected"] = CudaPlatform.is_fully_connected
    CudaPlatform.is_fully_connected = classmethod(
        lambda cls, physical_device_ids: True
    )


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
        if "is_fully_connected" in _ORIGINALS:
            from vllm.platforms.cuda import CudaPlatform

            CudaPlatform.is_fully_connected = _ORIGINALS.pop("is_fully_connected")
        if "offload_group_ids" in _ORIGINALS:
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
                config as _oc,
            )

            _oc.get_offloading_group_ids = _ORIGINALS.pop("offload_group_ids")[1]
        _INSTALLED = False
