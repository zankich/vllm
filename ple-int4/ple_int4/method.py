"""INT4 PLE embedding method + pinned-host backend for Qwen3.8-Flash-Next.

Subclasses and patches live entirely in this package; the vLLM tree is never
edited. Method semantics mirror Qwen4ExpPLEFp8EmbeddingMethod
(vllm/models/qwen4_exp/nvidia/ngram_embedding.py) with three deltas:

  - weight is int32 [rows, dim // 8] (8 packed symmetric int4 codes per word)
  - weight_scale is fp16 [rows, dim // group_size], per-group, pinned-host
  - dequantize() is an identity cast: the Triton lookup kernel already emits
    bf16 rows (global scale folded into group scales at pack time), so
    everything downstream of _lookup sees activation dtype directly.
"""
from __future__ import annotations

import torch

from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from ple_int4.kernel import lookup_ple_int4_from_pinned

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        scale = layer.weight_scale.data.to(torch.float32)
        if not torch.isfinite(scale).all():
            raise ValueError("int4 PLE checkpoint has non-finite group scales")

    def dequantize(self, layer, embeddings: torch.Tensor, output_dtype: torch.dtype):
        # The lookup kernel already emitted bf16; this is an identity cast so
        # the base-class contract (which expects storage-dtype rows) holds.
        return embeddings.to(output_dtype)

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "int4 PLE is pinned-host only; the resident device backend is not supported"
        )


class Qwen4ExpPLEPinnedHostInt4Embedding(Qwen4ExpPLEPinnedHostEmbedding):
    """Pinned-host PLE backend whose UVA lookup fuses int4 dequant.

    Constructed for both int4 and stock checkpoints (the module-global rebind
    in ple_int4.install swaps this class in unconditionally when the plugin
    loads). For a non-int4 weight the constructor changes nothing, so stock
    FP8/BF16 serving is byte-identical to upstream behavior.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.weight.dtype != torch.int32:
            return
        n_groups = int(self.weight_scale.shape[1])
        embedding_dim = self.weight.shape[1] * 8
        if n_groups * GROUP_SIZE != embedding_dim:
            raise ValueError(
                f"scale groups {n_groups} x {GROUP_SIZE} != embedding dim {embedding_dim}"
            )
        self._group_size = GROUP_SIZE
        # Second UVA view over the pinned per-group scales.
        self._uva_scale = get_accelerator_view_from_cpu_tensor(self.weight_scale)
        # The stock constructor sized the prefetch buffer to weight.dtype
        # (int32 here); the int4 lookup emits bf16, so reallocate.
        self._prefetch_buffer = torch.empty(
            self._prefetch_buffer.shape,
            dtype=torch.bfloat16,
            device=self._prefetch_buffer.device,
        )

    def _lookup(self, input_ids: torch.Tensor, output: torch.Tensor | None = None):
        """Full replacement: the stock _lookup validates output dtype against
        the (int32) weight dtype, which the bf16 dequant-fused output violates
        by design. Emits bf16 rows; out-of-shard rows are exact +0.0."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape, dtype=torch.bfloat16, device=input_ids.device
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != torch.bfloat16
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE int4 prefetch output must match the input shape, bf16 dtype, "
                "and input device"
            )
        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            lookup_ple_int4_from_pinned(
                self._uva_weight,
                self._uva_scale,
                flat_ids,
                output,
                vocab_start=int(self.shard_indices.org_vocab_start_index),
                vocab_end=int(self.shard_indices.org_vocab_end_index),
                group_size=self._group_size,
            )
        return output


def patched_load_weights(self, weights) -> set[str]:
    """Copy of Qwen4ExpNGramEmbedding.load_weights with two int4 deltas:

      - the shard shape check derives its column count from the destination
        weight (int32 packed columns for int4, elements otherwise), and
      - shard_{i}.weight_scale rows route through the same PLE shard loader
        (copy_ple_embedding_shard_ is rank-generic: [rows, n_groups] works).

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
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts

        matched_scale = name.startswith(shard_prefix) and name.endswith(".weight_scale")
        if name.startswith(shard_prefix) and (name.endswith(".weight") or matched_scale):
            suffix = ".weight_scale" if matched_scale else ".weight"
            shard_text = name[len(shard_prefix) : -len(suffix)]
            if not shard_text.isdigit():
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
            if matched_scale:
                param = embedding.weight_scale
            else:
                param = embedding.weight
            expected_shape = (expected_rows, param.shape[1])
            if tuple(loaded_weight.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for PLE embedding shard {shard_index}{suffix}: "
                    f"expected {expected_shape}, got {tuple(loaded_weight.shape)}"
                )
            param.weight_loader(
                param,
                loaded_weight,
                checkpoint_start=checkpoint_start,
            )
            loaded.add(f"ngram_embedding.{suffix}")
            continue
        regular_weights.append((name, loaded_weight))

    if regular_weights:
        loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
    return loaded
