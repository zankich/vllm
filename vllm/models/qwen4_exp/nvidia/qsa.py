# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import atexit
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


logger = init_logger(__name__)

_QSA_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_QSA_SUPPORTED_CACHE_DTYPES = ("auto", "bfloat16", *_QSA_FP8_CACHE_DTYPES)


def _is_qsa_fp8_cache_dtype(cache_dtype: str) -> bool:
    return cache_dtype in _QSA_FP8_CACHE_DTYPES


def _validated_qsa_fp8_dtype(cache_dtype: str) -> torch.dtype | None:
    """Return the platform E4M3 dtype after enforcing the validated HW gate."""

    if not _is_qsa_fp8_cache_dtype(cache_dtype):
        return None
    if not current_platform.is_cuda():
        raise ValueError(
            "Qwen4Exp QSA E4M3 KV cache is supported only on CUDA; "
            "ROCm and other platforms have not been validated"
        )
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() != 86:
        cap_str = (
            capability.as_version_str() if capability is not None else "unknown"
        )
        raise ValueError(
            "Qwen4Exp QSA E4M3 KV cache is validated only on SM86, but "
            f"{current_platform.get_device_name()} has compute capability "
            f"{cap_str}. Re-run with --kv-cache-dtype bfloat16."
        )
    return current_platform.fp8_dtype()


# --- FP8 K/V scale calibration collector ---------------------------------------
# Active ONLY when VLLM_QSA_KV_COLLECT names a writable directory. Records a
# GPU-side running absmax of the pre-quantization BF16 K/V per layer and
# dumps per-rank JSON for an offline max-merge into the VLLM_QSA_KV_SCALES
# sidecar. Must run with --enforce-eager: the collector mutates
# module-external state per step, which cudagraph capture would freeze.
_QSA_KV_COLLECT_ENV = "VLLM_QSA_KV_COLLECT"
_qsa_collect_dir = os.environ.get(_QSA_KV_COLLECT_ENV, "").strip()
_qsa_collect_state: dict[str, list[torch.Tensor]] = {}
_qsa_collect_calls = 0
_qsa_collect_rank: int | None = None


def _qsa_collect_dump() -> None:
    global _qsa_collect_rank
    if not _qsa_collect_state:
        return
    # Resolve the TP rank once and cache it: the atexit flush can run after
    # distributed teardown, where re-resolving would fall back to the pid and
    # write a differently-named duplicate next to the rank file.
    if _qsa_collect_rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            _qsa_collect_rank = get_tensor_model_parallel_rank()
        except Exception:
            _qsa_collect_rank = os.getpid()
    out = {
        name: {"k_absmax": float(st[0].item()), "v_absmax": float(st[1].item())}
        for name, st in _qsa_collect_state.items()
    }
    path = Path(_qsa_collect_dir) / f"qsa_absmax_rank{_qsa_collect_rank}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(path)


def _qsa_collect_absmax(
    layer_name: str, key: torch.Tensor, value: torch.Tensor
) -> None:
    global _qsa_collect_calls
    k_abs = key.detach().abs().amax().float()
    v_abs = value.detach().abs().amax().float()
    st = _qsa_collect_state.get(layer_name)
    if st is None:
        if not _qsa_collect_state:
            atexit.register(_qsa_collect_dump)
        _qsa_collect_state[layer_name] = [k_abs, v_abs]
    else:
        torch.maximum(st[0], k_abs, out=st[0])
        torch.maximum(st[1], v_abs, out=st[1])
    _qsa_collect_calls += 1
    if _qsa_collect_calls % 2000 == 0:
        _qsa_collect_dump()


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        # QSA never dispatches to a FlashAttention kernel; the base class
        # defers quantized dtypes to flash_attn_supports_kv_cache_dtype,
        # which is False for fp8 on sm_86. Answer from our own list.
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # FlashAttentionImpl.__init__ rejects any quantized KV dtype that FA
        # itself cannot read, and sm_86 has no FP8 FA kernel. Present "auto"
        # to the base class, then restore the real dtype. QSA never
        # dispatches to an FA kernel, and the only other kv_cache_dtype use
        # in that constructor is an SM90/FA4 dequant path that cannot
        # trigger here.
        requested_kv_cache_dtype = None
        if "kv_cache_dtype" in kwargs:
            requested_kv_cache_dtype = kwargs["kv_cache_dtype"]
            kwargs = {**kwargs, "kv_cache_dtype": "auto"}
        elif len(args) > 6:
            requested_kv_cache_dtype = args[6]
            args = (*args[:6], "auto", *args[7:])
        super().__init__(*args, **kwargs)
        if requested_kv_cache_dtype is not None:
            self.kv_cache_dtype = requested_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in _QSA_SUPPORTED_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 or static per-tensor E4M3 main "
                f"KV caches, not {self.kv_cache_dtype!r}"
            )
        self.qsa_fp8_dtype = _validated_qsa_fp8_dtype(self.kv_cache_dtype)
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 query")
        if key_cache.dtype != value_cache.dtype:
            raise TypeError("Qwen4Exp QSA K and V cache dtypes must match")
        if _is_qsa_fp8_cache_dtype(self.kv_cache_dtype):
            if self.qsa_fp8_dtype is None:
                raise RuntimeError("QSA FP8 dtype was not initialized")
            if key_cache.dtype not in (torch.uint8, self.qsa_fp8_dtype):
                raise NotImplementedError(
                    "Qwen4Exp QSA E4M3 cache storage must use raw uint8 or "
                    f"the platform FP8 dtype, not {key_cache.dtype}"
                )
        elif key_cache.dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen4Exp QSA BF16 mode requires a BF16 main KV cache"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        # Announce the effective scales once per layer: a deployment that
        # forgets the sidecar runs at the default 1.0, which is legitimate
        # but must never be silent.
        if _is_qsa_fp8_cache_dtype(self.kv_cache_dtype) and not getattr(
            self, "_qsa_scales_logged", False
        ):
            self._qsa_scales_logged = True
            logger.info(
                "Qwen4Exp QSA E4M3 KV active on %s: k_scale=%.6g v_scale=%.6g%s",
                getattr(layer, "layer_name", "<unnamed>"),
                float(layer._k_scale),
                float(layer._v_scale),
                ""
                if float(layer._k_scale) != 1.0 or float(layer._v_scale) != 1.0
                else "  (DEFAULT 1.0 -- no static calibration loaded)",
            )

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in _QSA_SUPPORTED_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 or static per-tensor E4M3 main "
                f"KV caches, not {cache_config.cache_dtype!r}"
            )
        qsa_fp8_dtype = _validated_qsa_fp8_dtype(cache_config.cache_dtype)
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None:
            raise NotImplementedError(
                "Qwen4Exp QSA does not implement kv_cache_scheme variants. "
                "Dynamic, per-head and per-token KV scales are unsupported; "
                "use static scalar per-layer K/V scales from "
                "load_qsa_static_kv_scales()."
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if _is_qsa_fp8_cache_dtype(self.kv_cache_dtype):
            if qsa_fp8_dtype is None:
                raise RuntimeError("QSA FP8 dtype was not initialized")
            if self.kv_cache_torch_dtype not in (torch.uint8, qsa_fp8_dtype):
                raise NotImplementedError(
                    "Qwen4Exp QSA E4M3 cache storage resolved to unsupported "
                    f"dtype {self.kv_cache_torch_dtype}"
                )
        elif self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen4Exp QSA BF16 mode requires BF16 cache storage"
            )
        # The checkpoint carries no KV scale scheme: these registered scalar
        # buffers stay 1.0 until load_qsa_static_kv_scales() replaces them
        # after weight load.
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        if _qsa_collect_dir:
            _qsa_collect_absmax(self.layer_name, key, value)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
        )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def load_qsa_static_kv_scales(
    layers: nn.Module | Mapping[str, nn.Module],
    sidecar_path: str | Path,
    *,
    strict: bool = True,
) -> list[str]:
    """Load static scalar QSA K/V scales into already-loaded model layers.

    Call this on every model worker after checkpoint weight loading finishes
    and before any profiling, warmup, or cudagraph capture. ``layers`` may be
    the loaded root module or ``compilation_config.static_forward_context``.

    The JSON schema is::

        {
          "model.layers.3.self_attn.attn": {
            "k_scale": 0.25,
            "v_scale": 2.0
          }
        }

    ``strict=False`` permits a deliberate subset; omitted layers keep the
    vLLM default scale of 1.0. Unknown names and non-scalar values are always
    errors.
    """

    path = Path(sidecar_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by layer name")

    if isinstance(layers, nn.Module):
        qsa_layers = {
            layer.layer_name: layer
            for layer in layers.modules()
            if isinstance(layer, Qwen4ExpQSAAttention)
        }
    else:
        qsa_layers = {
            name: layer
            for name, layer in layers.items()
            if isinstance(name, str) and isinstance(layer, Qwen4ExpQSAAttention)
        }

    if not qsa_layers:
        raise ValueError("No Qwen4Exp QSA attention layers were supplied")
    non_fp8_layers = sorted(
        name
        for name, layer in qsa_layers.items()
        if not _is_qsa_fp8_cache_dtype(layer.kv_cache_dtype)
    )
    if non_fp8_layers:
        raise ValueError(
            "Static QSA KV scales may only be injected into E4M3 layers; "
            f"non-FP8 layers: {non_fp8_layers}"
        )

    unknown = sorted(name for name in raw if name not in qsa_layers)
    if unknown:
        raise ValueError(f"Unknown QSA layer names in {path}: {unknown}")
    missing = sorted(name for name in qsa_layers if name not in raw)
    if strict and missing:
        raise ValueError(f"Missing QSA layer scales in {path}: {missing}")

    validated: dict[str, tuple[float, float]] = {}
    for layer_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{layer_name} scales must be a JSON object")
        if set(entry) != {"k_scale", "v_scale"}:
            raise ValueError(f"{layer_name} must contain exactly k_scale and v_scale")

        values: list[float] = []
        for field in ("k_scale", "v_scale"):
            value = entry[field]
            if type(value) not in (int, float):
                raise ValueError(f"{layer_name}.{field} must be a scalar JSON number")
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{layer_name}.{field} must be finite and positive")
            values.append(scale)

        layer = qsa_layers[layer_name]
        for field in ("_k_scale", "_v_scale"):
            tensor = getattr(layer, field, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
                raise ValueError(
                    f"{layer_name}.{field} must be an existing scalar tensor"
                )
        validated[layer_name] = (values[0], values[1])

    with torch.no_grad():
        for layer_name, (k_scale, v_scale) in validated.items():
            layer = qsa_layers[layer_name]
            layer._k_scale.fill_(k_scale)
            layer._v_scale.fill_(v_scale)

    return sorted(validated)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "load_qsa_static_kv_scales",
]
