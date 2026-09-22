"""PLE lookup probe: row-frequency histogram and kernel timing.

Answers the two questions a VRAM hot-row cache depends on, without
changing serving behavior:

  1. Coverage — what fraction of lookups the top-N hottest rows serve,
     for N at each VRAM budget. The table is 320,001,536 rows at 90 B
     (160 int4 values + 5 fp16 group scales), 6.71 GiB per TP4 rank, so
     512 MiB of VRAM holds ~7.5% of a rank's rows. A cache only pays if
     3-gram traffic is Zipfian enough for that slice to serve most hits.
  2. Cost — what fraction of step time the lookup kernel occupies. Even
     perfect caching is worthless if the kernel is noise.

Off unless PLE_INT4_PROBE=1. Enabling costs a device-to-host copy of the
id tensor per lookup, so it is a measurement tool, not a serving mode.

Dump with PLE_INT4_PROBE_DUMP=/path/prefix (per-rank files are written
on interpreter exit, or on demand via dump()).
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import Counter


class LookupProbe:
    """Counts row-id frequencies and kernel wall time per rank."""

    def __init__(self, dump_prefix: str | None = None) -> None:
        self._counter: Counter[int] = Counter()
        self._lock = threading.Lock()
        self._lookups = 0
        self._rows_requested = 0
        self._dump_prefix = dump_prefix
        self._started = time.monotonic()

    def record(self, flat_ids, vocab_start: int, vocab_end: int) -> None:
        """Record one launch's row ids.

        Ids outside [vocab_start, vocab_end) belong to other ranks' shards
        and are skipped by the kernel, so they are not counted here either.
        """
        try:
            ids = flat_ids.detach().to("cpu", non_blocking=False).tolist()
        except Exception:
            return
        with self._lock:
            self._lookups += 1
            for rid in ids:
                if vocab_start <= rid < vocab_end:
                    self._rows_requested += 1
                    self._counter[rid] += 1

    def coverage(self, budgets_mib: tuple[int, ...] = (64, 128, 256, 512, 1024)):
        """Fraction of lookups served by the top-N rows at each budget."""
        bytes_per_row = 90
        with self._lock:
            total = sum(self._counter.values())
            if not total:
                return []
            ordered = sorted(self._counter.values(), reverse=True)
        out = []
        for mib in budgets_mib:
            n_rows = mib * (2**20) // bytes_per_row
            covered = sum(ordered[:n_rows])
            out.append((mib, n_rows, covered / total))
        return out

    def summary(self) -> str:
        with self._lock:
            distinct = len(self._counter)
            total = self._rows_requested
            lookups = self._lookups
            elapsed = time.monotonic() - self._started
        lines = [
            f"ple-probe: lookups={lookups} rows_requested={total} "
            f"distinct_rows={distinct} elapsed={elapsed:.1f}s",
        ]
        for mib, n_rows, frac in self.coverage():
            lines.append(
                f"  cache {mib:>5} MiB ({n_rows:>9,} rows): "
                f"{100 * frac:6.2f}% of lookups served"
            )
        return "\n".join(lines)

    def dump(self, path: str | None = None) -> None:
        target = path or self._dump_prefix
        if not target:
            return
        rank = os.environ.get("VLLM_DP_RANK", "") or str(os.getpid())
        with self._lock:
            items = self._counter.most_common()
        with open(f"{target}.rank{rank}.tsv", "w") as fh:
            fh.write(f"# {self.summary()}\n")
            for rid, count in items:
                fh.write(f"{rid}\t{count}\n")


def _maybe_build() -> LookupProbe | None:
    if os.environ.get("PLE_INT4_PROBE", "0") != "1":
        return None
    probe = LookupProbe(os.environ.get("PLE_INT4_PROBE_DUMP") or None)
    atexit.register(probe.dump)
    return probe
