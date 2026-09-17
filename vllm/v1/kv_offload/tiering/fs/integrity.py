# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local integrity records for the fs offload tier.

The tier's payload files are content-addressed by path only: nothing binds a
file's bytes to the key they were stored under, so full-length but wrong bytes
restore without any error (observed 2026-09-16/17: tier storage wiped or
damaged under a live engine, later restores attended to garbage from token 0).
garbage from token 0). Each store writes a sidecar ``<path>.meta`` naming the
key identity and a checksum of (key, payload); loads verify before the block
is trusted. Any mismatch, missing record, or storage-identity change degrades
to a miss and a full recompute — never a partial trust of the rest.
"""

import hashlib
import os
import struct

SIDECAR_SUFFIX = ".meta"

_MAGIC = b"KVMI"
_VERSION = 1
_HEADER = struct.Struct("<4sBH16s")  # magic, version, key length, checksum


def sidecar_path(path: str) -> str:
    return path + SIDECAR_SUFFIX


def block_checksum(key: bytes, payload: bytes | memoryview) -> bytes:
    """Bind a block's payload to the key it is stored under."""
    h = hashlib.sha256()
    h.update(len(key).to_bytes(8, "big"))
    h.update(key)
    h.update(payload)
    return h.digest()[:16]


def write_sidecar(path: str, key: bytes, payload: bytes | memoryview) -> None:
    """Atomically write the integrity record for a stored payload file."""
    record = _HEADER.pack(_MAGIC, _VERSION, len(key), block_checksum(key, payload))
    record += key
    tmp = sidecar_path(path) + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(record)
    os.replace(tmp, sidecar_path(path))


def read_sidecar(path: str) -> tuple[bytes, bytes] | None:
    """Return ``(key identity, checksum)``; None when absent or unusable."""
    try:
        with open(sidecar_path(path), "rb") as f:
            record = f.read()
    except OSError:
        return None
    if len(record) < _HEADER.size:
        return None
    magic, version, key_len, checksum = _HEADER.unpack_from(record)
    if magic != _MAGIC or version != _VERSION:
        return None
    key = record[_HEADER.size : _HEADER.size + key_len]
    if len(key) != key_len:
        return None
    return key, checksum


def remove_block(path: str) -> None:
    """Best-effort removal of a rejected payload and its record."""
    for p in (path, sidecar_path(path)):
        try:
            os.remove(p)
        except OSError:
            pass
