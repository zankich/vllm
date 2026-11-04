# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""INT4 PLE embedding method + pinned-host backend for Qwen3.8-Flash-Next.

Subclasses and patches live entirely in this package; the vLLM tree is never
edited. The method rides upstream's pinned-host lookup seam
(vllm/models/qwen4_exp/common/ngram_embedding.py): ``lookup_dtype`` names the
dtype the pinned lookup writes (the activation dtype, like the NVFP4 method)
and ``lookup_from_pinned`` swaps in the int4 fused-dequant UVA kernel, so the
stock ``_lookup`` validation, the lazy prefetch buffer and the side-stream
prefetch machinery all run unmodified. Deltas from the stock method:

  - weight is int32 [rows, dim // 8] (8 packed symmetric int4 codes per word)
  - weight_scale is fp16 [rows, dim // group_size], per-group, pinned-host
  - the lookup fuses int4 dequant, so dequantize() is an identity cast
"""

from __future__ import annotations

import torch

from ple_int4.kernel import lookup_ple_int4_from_pinned
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

GROUP_SIZE = 32  # values per scale group; 5 groups on the 160-wide rows


class Qwen4ExpPLEInt4EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Pinned-host int4 PLE storage with fused-dequant UVA lookup."""

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

        if not isinstance(layer, Qwen4ExpPLEPinnedHostInt4Embedding):
            raise RuntimeError(
                "ple_int4: the int4 PLE method was constructed against a layer "
                "that is not the int4 pinned-host embedding "
                "(Qwen4ExpPLEPinnedHostInt4Embedding). The stock pinned-host "
                "class is resolved through module globals the plugin does not "
                "rebind (the shared AMD PLE path, and the resident-device "
                "backend when engram cpu_offload is off), so this plugin's "
                "from_quant_config wrapper can hand it the int4 method. The "
                "stock byte-copy lookup would then index int32-packed rows "
                "with the full row stride and read out of bounds, silently. "
                "Refusing the mismatch."
            )
        del input_size, output_size, params_dtype
        if input_size_per_partition % 8 or input_size_per_partition % GROUP_SIZE:
            raise ValueError(
                f"PLE embedding dim {input_size_per_partition} not divisible by "
                f"8 and group size {GROUP_SIZE}"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition // 8,
                torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition // GROUP_SIZE,
                torch.float16,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        if layer.weight_scale.is_pinned():
            # UVA view over the pinned per-group scales, used by the lookup.
            self._uva_scale = get_accelerator_view_from_cpu_tensor(weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        scale = layer.weight_scale.data.to(torch.float32)
        if not torch.isfinite(scale).all():
            raise ValueError("int4 PLE checkpoint has non-finite group scales")

    def lookup_dtype(self, layer: torch.nn.Module) -> torch.dtype:
        # The int4 lookup decodes rows straight to the activation dtype.
        return layer.params_dtype

    def lookup_from_pinned(
        self,
        layer: Qwen4ExpPLEPinnedHostEmbedding,
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        lookup_ple_int4_from_pinned(
            layer._uva_weight,
            self._uva_scale,
            ids,
            output,
            vocab_start=int(layer.shard_indices.org_vocab_start_index),
            vocab_end=int(layer.shard_indices.org_vocab_end_index),
            group_size=GROUP_SIZE,
        )

    def dequantize(
        self,
        layer: torch.nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        # The lookup already decoded the rows to the activation dtype.
        return embeddings.to(output_dtype)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "int4 PLE is pinned-host only; the resident device backend is not supported"
        )


class Qwen4ExpPLEPinnedHostInt4Embedding(Qwen4ExpPLEPinnedHostEmbedding):
    """Marker subclass selecting the int4 method's storage at construction.

    The base class owns the lookup machinery: ``_lookup`` validates and
    allocates at the embedding method's ``lookup_dtype``, the prefetch buffer
    and side stream allocate lazily at first use, and the pinned weight's UVA
    view is taken in ``__init__``. No override is needed here; the class
    exists so config-driven binding can construct the int4 storage and the
    int4 method can refuse any other layer type.
    """


def patched_load_weights(self, weights) -> set[str]:
    """Copy of Qwen4ExpNGramEmbedding.load_weights with the packed-table
    shard routing extended to the int4 method.

    Upstream routes ``shard_{i}.weight_scale`` tensors through the PLE shard
    loader (and derives the shard shape check from the destination
    parameter) only for NVFP4 tables. The int4 table has the same property:
    its per-group scales are checkpoint-sharded like its rows, and its
    packed int32 words must never be cast. Unsharded packed tables are
    refused loudly rather than falling through to the generic loader.

    Hash buffers and regular weights behave exactly as upstream. Authored
    against the pinned nightly's file; install() asserts the source hash.
    """
    persistent_buffers = {
        "layer_multipliers": self.layer_multipliers,
        "ngram_heads_offsets": self.ngram_heads_offsets,
        "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
    }
    loaded: set[str] = set()
    regular_weights: list = []
    shard_prefix = "ngram_embedding.shard_"

    for name, loaded_weight in weights:
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
            continue
        if name in persistent_buffers:
            buffer = persistent_buffers[name]
            if buffer.shape != loaded_weight.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected "
                    f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                )
            buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
            loaded.add(name)
            continue

        embedding = self.ngram_embedding
        method = getattr(embedding, "embedding_method", None)
        # Packed-table methods shard their scales like their rows.
        packed_method = (
            method
            if isinstance(
                method,
                (Qwen4ExpPLENvFp4EmbeddingMethod, Qwen4ExpPLEInt4EmbeddingMethod),
            )
            else None
        )
        shard_parameters = (
            ("weight", "weight_scale") if packed_method is not None else ("weight",)
        )
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts

        if name.startswith(shard_prefix):
            shard_text, _, suffix = name[len(shard_prefix) :].partition(".")
            if not shard_text.isdigit() or suffix not in shard_parameters:
                regular_weights.append((name, loaded_weight))
                continue
            shard_index = int(shard_text)
            if shard_index >= self.split_ngram_parts:
                raise ValueError(
                    f"PLE embedding shard index {shard_index} exceeds "
                    f"split_ngram_parts={self.split_ngram_parts}"
                )
            checkpoint_start = shard_index * shard_size
            expected_rows = max(
                0,
                min(shard_size, embedding.org_vocab_size - checkpoint_start),
            )
            parameter = getattr(embedding, suffix)
            expected_shape = (expected_rows, parameter.shape[1])
            if tuple(loaded_weight.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for PLE embedding shard {shard_index} "
                    f"{suffix}: expected {expected_shape}, got "
                    f"{tuple(loaded_weight.shape)}"
                )
            # A cast would silently reinterpret packed int4/NVFP4 bytes.
            if packed_method is not None and loaded_weight.dtype != parameter.dtype:
                raise ValueError(
                    f"int4/NVFP4 PLE shard {shard_index} {suffix} requires "
                    f"{parameter.dtype}, got {loaded_weight.dtype}"
                )
            parameter.weight_loader(
                parameter,
                loaded_weight,
                checkpoint_start=checkpoint_start,
            )
            loaded.add(f"ngram_embedding.{suffix}")
            continue
        if packed_method is not None and name in (
            "ngram_embedding.weight",
            "ngram_embedding.weight_scale",
        ):
            raise ValueError(
                f"int4/NVFP4 PLE tables must be stored as {shard_prefix}<i>.weight "
                f"and {shard_prefix}<i>.weight_scale shards, got unsharded {name}"
            )
        regular_weights.append((name, loaded_weight))

    if regular_weights:
        loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
    return loaded
