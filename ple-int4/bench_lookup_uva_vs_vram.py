"""Cost bound for the PLE hot-row cache: UVA (pinned host) vs VRAM.

Runs the real int4 lookup kernel at decode-shaped batch sizes against
(a) the pinned-host table it uses in production and (b) an identical
table resident in VRAM. The delta is the entire prize a hot-row cache
could win, before any hit-rate discount.

Standalone: no engine, no weights. Allocates a small synthetic table
(the real one is 6.71 GiB/rank) with production row geometry — 20 int32
words + 5 fp16 scales = 160 int4 values per row.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple_int4.kernel import lookup_ple_int4_from_pinned  # noqa: E402

WORDS_PER_ROW = 20
N_GROUPS = 5
GROUP_SIZE = 32
EMBED_DIM = WORDS_PER_ROW * 8  # 160 int4 values


def _bench(w, s, ids, out, iters: int) -> float:
    for _ in range(3):
        lookup_ple_int4_from_pinned(
            w, s, ids, out,
            vocab_start=0, vocab_end=w.shape[0], group_size=GROUP_SIZE,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        lookup_ple_int4_from_pinned(
            w, s, ids, out,
            vocab_start=0, vocab_end=w.shape[0], group_size=GROUP_SIZE,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds per launch


def main() -> None:
    rows = int(os.environ.get("PROBE_ROWS", 2_000_000))
    iters = int(os.environ.get("PROBE_ITERS", 200))
    torch.cuda.init()

    print(f"table rows={rows:,} ({rows * 90 / 2**20:.0f} MiB at 90 B/row)")
    pinned_w = torch.empty((rows, WORDS_PER_ROW), dtype=torch.int32, pin_memory=True)
    pinned_s = torch.ones((rows, N_GROUPS), dtype=torch.float16, pin_memory=True)
    pinned_w.random_(0, 2**31 - 1)

    vram_w = pinned_w.to("cuda")
    vram_s = pinned_s.to("cuda")

    print(f"{'batch':>8} {'uva_us':>10} {'vram_us':>10} {'delta_us':>10} {'speedup':>8}")
    for batch in (4, 8, 12, 24, 48, 96):
        ids = torch.randint(0, rows, (batch,), dtype=torch.int64, device="cuda")
        out = torch.empty((batch, EMBED_DIM), dtype=torch.bfloat16, device="cuda")
        uva_us = _bench(pinned_w, pinned_s, ids, out, iters)
        vram_us = _bench(vram_w, vram_s, ids, out, iters)
        print(
            f"{batch:>8} {uva_us:>10.1f} {vram_us:>10.1f} "
            f"{uva_us - vram_us:>10.1f} {uva_us / vram_us:>7.2f}x"
        )


if __name__ == "__main__":
    main()
