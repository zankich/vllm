"""Streaming FP8 -> INT4 repack of the Qwen3.8-Flash-Next PLE n-gram table.

Library surface (used by the CLI, the always-on self-check, and the G3f
synthetic gate):
    pack_table_to_int4(tensor, group_size, global_scale) -> (words, scales)
    decode_int4(words, scales, group_size) -> fp32 tensor

Layout contract (must match ple_int4.kernel):
    words  int32 [rows, dim // 8]     8 little-endian nibbles per int32,
                                       element i of each 8-group in bits 4i
    scales fp16 [rows, dim // group]  per-group symmetric scales, with the
                                       FP8 global scale folded in; zero groups
                                       carry the sentinel scale 1.0 and all-
                                       zero codes
    value  = (code - 8) * scale

The packer computes codes against the fp16-rounded scale (encode/decode share
one scale value), so the round-trip residual is bounded by half a quant step
plus the fp32 division epsilon, well inside s * (0.5 + 1e-3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field

import torch
from safetensors.torch import safe_open, save_file

PLE_SHARD_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
GLOBAL_SCALE_NAME = f"{PLE_SHARD_PREFIX}.weight_scale"
ROW_BLOCK = 1 << 19  # 512K rows ~ 160 MB fp8 / 320 MB fp32 work arrays


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def derive_ple_geometry(text_config: dict) -> tuple[list[int], int]:
    """Port of Qwen4ExpNGramEmbedding._make_vocab_layout plus the padding at
    ngram_embedding.py:692-693: per-head sizes and the PADDED total row count
    (vLLM builds the embedding with padded_vocab_size, so the checkpoint's
    shards partition the padded table and every shard is full-size)."""
    base = int(text_config["ngram_vocab_size_base"])
    heads = (int(text_config["ngram_size"]) - 1) * int(text_config["heads_per_ngram"])
    sizes: list[int] = []
    offset = 0
    for local_head in range(heads):
        size = _nth_prime_after(base - 1, local_head + 1)
        sizes.append(size)
        offset += size
    divisor = int(text_config.get("make_ngram_vocab_size_divisible_by", 1))
    padded = ((offset + divisor - 1) // divisor) * divisor
    return sizes, padded


def pack_table_to_int4(
    tensor: torch.Tensor,
    group_size: int = 32,
    global_scale: float = 1.0,
    block_rows: int = ROW_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack any float tensor [rows, dim] to (words int32, scales fp16).

    Symmetric int4, codes clamp(round(v / s), -8, 7) stored as code + 8;
    scales = maxabs * global_scale / 7 rounded to fp16, computed per group.
    Zero groups take sentinel scale 1.0 with zero codes (decodes to +0.0
    exactly). Codes are computed against the fp16-rounded scale.
    """
    if tensor.dim() != 2:
        raise ValueError(f"expected 2-D table, got {tuple(tensor.shape)}")
    rows, dim = tensor.shape
    if dim % 8 or dim % group_size:
        raise ValueError(f"dim {dim} not divisible by 8 and group_size {group_size}")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"expected float dtype, got {tensor.dtype}")

    n_groups = dim // group_size
    words = torch.empty(rows, dim // 8, dtype=torch.int32)
    scales_out = torch.empty(rows, n_groups, dtype=torch.float16)

    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        # Work on the global-scale-folded reference: decode = q * s must
        # approximate tensor * global_scale (the FP8 table's real values).
        vs = tensor[start:stop].to(torch.float32) * global_scale
        g = vs.abs().reshape(stop - start, n_groups, group_size)
        maxabs = g.amax(dim=-1)
        s = (maxabs / 7.0).to(torch.float16)
        sentinels = ~torch.isfinite(s) | (s == 0)
        if not torch.isfinite(s[~sentinels]).all():
            raise ValueError("non-finite group scale after fp16 rounding")
        s = torch.where(sentinels, torch.ones_like(s), s)
        # Broadcast the stored (fp16-rounded) scale for code computation so
        # encode and decode share one scale value exactly.
        sb = s.to(torch.float32).repeat_interleave(group_size, dim=-1)
        q = torch.clamp(torch.round(vs / sb), -8.0, 7.0).to(torch.int8)
        if sentinels.any():
            q = torch.where(
                sentinels.repeat_interleave(group_size, dim=-1),
                torch.zeros_like(q),
                q,
            )
        nib = (q.to(torch.int32) + 8) & 0xF
        words[start:stop] = _pack_nibbles(nib)
        scales_out[start:stop] = s
    return words, scales_out


def _pack_nibbles(nib: torch.Tensor) -> torch.Tensor:
    """nib [rows, dim] int32 in 0..15 -> words [rows, dim//8] little-endian."""
    rows, dim = nib.shape
    grouped = nib.reshape(rows, dim // 8, 8)
    words = torch.zeros(rows, dim // 8, dtype=torch.int32)
    for i in range(8):
        words |= grouped[:, :, i] << (4 * i)
    return words


def _unpack_nibbles(words: torch.Tensor) -> torch.Tensor:
    """words [rows, dim//8] -> nib [rows, dim], element i of each 8 in slot i."""
    rows, n_words = words.shape
    out = torch.empty(rows, n_words * 8, dtype=torch.int32)
    for i in range(8):
        out[:, i::8] = (words >> (4 * i)) & 0xF
    return out


def decode_int4(words: torch.Tensor, scales: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Reference decode to fp32; mirrors the Triton kernel exactly."""
    codes = _unpack_nibbles(words).to(torch.float32) - 8.0
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=-1)
    return codes * s


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    src: str
    out: str
    group_size: int
    global_scale: float
    total_rows: int
    head_dim: int
    shards: list = field(default_factory=list)
    subnormal_scales: int = 0
    sampled_rows: int = 0
    max_residual_rel: float = 0.0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def convert(src_dir: str, out_dir: str, group_size: int = 32, check_rows: int = 4096) -> Manifest:
    """Repack the halt95 FP8 PLE shards into int4 and assemble a servable dir."""
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    ple_shards = {
        name: fname
        for name, fname in weight_map.items()
        if name.startswith(f"{PLE_SHARD_PREFIX}.shard_") and name.endswith(".weight")
    }
    if not ple_shards:
        raise ValueError(f"no PLE shard tensors found under {PLE_SHARD_PREFIX}.shard_*")

    with open(os.path.join(src_dir, "config.json")) as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    head_sizes, total_rows = derive_ple_geometry(text_config)
    n_heads = len(head_sizes)
    head_dim = int(text_config["ple_embed_dim"]) // n_heads
    if head_dim % 8 or head_dim % group_size:
        raise ValueError(f"head_dim {head_dim} incompatible with group_size {group_size}")
    # Shards partition the whole concatenated table into split_ngram_parts
    # pieces (see Qwen4ExpNGramEmbedding.load_weights), not per head.
    split_parts = int(text_config.get("split_ngram_parts", 512))
    shard_size = (total_rows + split_parts - 1) // split_parts
    if len(ple_shards) != split_parts:
        raise ValueError(f"{len(ple_shards)} shard tensors != split_ngram_parts {split_parts}")

    global_scale = 1.0
    if GLOBAL_SCALE_NAME in weight_map:
        fname = weight_map[GLOBAL_SCALE_NAME]
        with safe_open(os.path.join(src_dir, fname), framework="pt") as f:
            global_scale = float(f.get_tensor(GLOBAL_SCALE_NAME).item())

    out_files: list[str] = []
    out_index: dict[str, str] = {}
    manifest = Manifest(src=src_dir, out=out_dir, group_size=group_size,
                        global_scale=global_scale, total_rows=total_rows, head_dim=head_dim)
    rng = torch.Generator().manual_seed(20260915)

    # Group shards by their source file so the output mirrors the 10-file layout.
    by_src_file: dict[str, list[str]] = {}
    for name, fname in sorted(ple_shards.items()):
        by_src_file.setdefault(fname, []).append(name)

    for file_no, (src_file, names) in enumerate(sorted(by_src_file.items())):
        out_file = f"model-pleint4-{file_no:05d}-of-{len(by_src_file):05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            shard_index = int(name[len(f"{PLE_SHARD_PREFIX}.shard_") : -len(".weight")])
            expected_rows = max(0, min(shard_size, total_rows - shard_index * shard_size))
            with safe_open(os.path.join(src_dir, src_file), framework="pt") as f:
                table = f.get_tensor(name)
            if tuple(table.shape) != (expected_rows, head_dim):
                raise ValueError(
                    f"{name}: expected {(expected_rows, head_dim)}, got {tuple(table.shape)}"
                )
            if table.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32):
                raise ValueError(f"{name}: unexpected dtype {table.dtype}")
            words, scales = pack_table_to_int4(table, group_size, global_scale)
            base = name[: -len(".weight")]
            tensors[f"{base}.weight"] = words
            tensors[f"{base}.weight_scale"] = scales
            out_index[f"{base}.weight"] = out_file
            out_index[f"{base}.weight_scale"] = out_file

            # Always-on self-check on sampled rows (independent decode path).
            rows = words.shape[0]
            n = min(check_rows, rows)
            pick = torch.randperm(rows, generator=rng)[:n]
            ref = table[pick].to(torch.float32) * global_scale
            dec = decode_int4(words[pick], scales[pick], group_size)
            s = scales[pick].to(torch.float32).repeat_interleave(group_size, dim=-1)
            bound = s * (0.5 + 1e-3)
            residual = (ref - dec).abs()
            rel = (residual / (s + 1e-30)).max().item() if rows else 0.0
            if (residual > bound).any():
                raise ValueError(f"{name}: {int((residual > bound).sum())} sampled values exceed round-trip bound")
            zero_rows = ref.abs().amax(dim=-1) == 0
            if zero_rows.any() and not torch.all(dec[zero_rows] == 0):
                raise ValueError(f"{name}: zero rows do not decode to exact 0")
            if not torch.isfinite(scales[pick].to(torch.float32)).all():
                raise ValueError(f"{name}: non-finite scales")
            subnormal = int(((scales[pick].to(torch.float32) > 0) & (scales[pick].to(torch.float32) < 6.1e-5)).sum())
            codes_back = torch.clamp(torch.round(ref / s), -8.0, 7.0) + 8
            if not torch.equal(codes_back.long(), _unpack_nibbles(words[pick]).long()):
                raise ValueError(f"{name}: decode does not reproduce stored codes")
            manifest.shards.append({"tensor": name, "rows": rows, "subnormal_scales": subnormal,
                                    "sha256_words": None, "max_rel_residual": rel})
            manifest.subnormal_scales += subnormal
            manifest.sampled_rows += n
            manifest.max_residual_rel = max(manifest.max_residual_rel, rel)
            del table, words, scales, ref, dec
        os.makedirs(out_dir, exist_ok=True)
        save_file(tensors, os.path.join(out_dir, out_file))
        out_files.append(out_file)
        del tensors  # do not retain ~2.9 GB per file across the run

    os.makedirs(out_dir, exist_ok=True)

    # Symlink every non-PLE file from the source; absolute paths (compose bind-mounts /models).
    ple_src_files = set(by_src_file)
    for entry in os.listdir(src_dir):
        src_path = os.path.join(src_dir, entry)
        if entry in ("model.safetensors.index.json", "config.json") or entry in ple_src_files:
            continue
        if not os.path.isfile(src_path):
            continue
        link = os.path.join(out_dir, entry)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src_path), link)
        if not os.path.exists(link):
            raise ValueError(f"broken symlink for {entry}")

    new_config = json.loads(json.dumps(config))
    tc = new_config.get("text_config", new_config)
    tc["ple_embedding_dtype"] = "int4"
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2, sort_keys=True)

    # Merge: keep every original (non-PLE) entry so the rest of the model
    # still loads; replace PLE shard entries with the int4 outputs and drop
    # the stale global scale. A load log of "10/10 shards" means this merge
    # was skipped and the model body never loaded.
    dropped = {n for n in weight_map if n.startswith(f"{PLE_SHARD_PREFIX}.shard_")}
    dropped.add(GLOBAL_SCALE_NAME)
    merged = {n: f for n, f in weight_map.items() if n not in dropped}
    overlap = set(merged) & set(out_index)
    if overlap:
        raise ValueError(f"index collision on non-PLE names: {sorted(overlap)[:3]}")
    merged.update(out_index)
    lost = {n for n in weight_map if n not in merged and n not in dropped}
    if lost:
        raise ValueError(f"non-PLE tensors lost from index: {sorted(lost)[:5]}")
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": dict(sorted(merged.items())),
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=True)

    file_digests = {
        out_file: _sha256(os.path.join(out_dir, out_file)) for out_file in out_files
    }
    for s in manifest.shards:
        s["file"] = out_index.get(s["tensor"])
    with open(os.path.join(out_dir, "ple-int4-manifest.json"), "w") as f:
        json.dump({"manifest": json.loads(manifest.to_json()), "file_sha256": file_digests},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="halt95 W4A16-Merlin checkpoint dir")
    ap.add_argument("--out", required=True, help="output int4-PLE checkpoint dir")
    ap.add_argument("--group-size", type=int, default=32, choices=(32, 160))
    ap.add_argument("--check-rows", type=int, default=4096)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = convert(args.src, args.out, args.group_size, args.check_rows)
    print(m.to_json())


if __name__ == "__main__":
    main()
