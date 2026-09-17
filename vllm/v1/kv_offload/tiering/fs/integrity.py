# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local integrity records for the fs offload tier.

The tier's payload files are content-addressed by path only: nothing binds a
file's bytes to the key they were stored under, so full-length but wrong bytes
restore without any error (observed 2026-09-16/17: tier storage wiped or
damaged under a live engine, later restores attended to garbage from token 0).
Each store records the key identity and a checksum of
(key, payload) in a ``user.*`` extended attribute on the payload file; loads
verify before the block is trusted. Any mismatch, missing record, or
storage-identity change degrades to a miss and a full recompute — never a
partial trust of the rest.

The record travels with the inode: replacement or deletion of the payload
drops it atomically, and no second file exists for the pruner to orphan.
Linux user.* xattrs only; the tier probes support at construction and fails
loud rather than silently running as a 100% miss.
"""

import errno
import hashlib
import os
import struct

XATTR_NAME = "user.vllm_kv_integrity"

_MAGIC = b"KVMI"
_VERSION = 1
_HEADER = struct.Struct("<4sBH16s")  # magic, version, key length, checksum


def probe_xattr(directory: str) -> None:
    """Raise unless user.* xattrs round-trip in *directory*.

    Mirrors ``probe_o_direct``: some surfaces (pre-6.12 tmpfs, many NFS
    servers) reject user.* attributes, which would otherwise turn every
    record write into a silent miss-everything tier.
    """
    path = os.path.join(directory, f".xattr_probe_{os.getpid()}")
    try:
        with open(path, "wb"):
            pass
        try:
            record = _HEADER.pack(_MAGIC, _VERSION, 0, b"\x00" * 16)
            os.setxattr(path, XATTR_NAME, record)
            if os.getxattr(path, XATTR_NAME) != record:
                raise OSError("xattr round-trip mismatch")
            os.removexattr(path, XATTR_NAME)
        finally:
            os.remove(path)
    except OSError as exc:
        if exc.errno in (errno.EPERM, errno.EACCES):
            raise
        raise ValueError(
            f"KV offload fs tier requires user.* xattr support at "
            f"'{directory}' (got {exc}); refusing to start as a silent "
            f"100% miss"
        ) from exc


def block_checksum(key: bytes, payload: bytes | memoryview) -> bytes:
    """Bind a block's payload to the key it is stored under."""
    h = hashlib.sha256()
    h.update(len(key).to_bytes(8, "big"))
    h.update(key)
    h.update(payload)
    return h.digest()[:16]


def write_record(path: str, key: bytes, payload: bytes | memoryview) -> None:
    """Attach the integrity record to a stored payload file.

    Runs after the payload is published (the C batch store owns the rename),
    so a crash in between leaves a record-less payload the load path rejects,
    removes, and the next store rewrites.
    """
    record = _HEADER.pack(_MAGIC, _VERSION, len(key), block_checksum(key, payload))
    record += key
    os.setxattr(path, XATTR_NAME, record)


def read_record(path: str) -> tuple[bytes, bytes] | None:
    """Return ``(key identity, checksum)``; None when absent or unusable.

    Only ENODATA (payload present, record absent) and a present-but-invalid
    record count as unverifiable. Any other OSError propagates: ELOOP, EACCES
    or EIO are transient host conditions, and the caller must fail the load
    without touching the file, not treat it as corruption.
    """
    try:
        record = os.getxattr(path, XATTR_NAME)
    except OSError as exc:
        if exc.errno == errno.ENODATA:
            return None
        raise
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
    """Best-effort removal of a rejected payload (its record dies with it)."""
    try:
        os.remove(path)
    except OSError:
        pass
