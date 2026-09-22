"""Decode/prefill profiling harness for the int4-PLE Flash-Next serve.

Env knobs (docker compose run -e K=V flashnext-int4ple):
  EAGER=1|0   enforce_eager (default 0 = graphs on)
  MTPK=int    MTP speculative tokens (default 3; 0 = off)
  CAPTURES    comma list of cudagraph capture sizes (default: auto, or
              [1, seqs*(K+1)] when MTPK is set)
  BATCHTOK    max_num_batched_tokens (default 2048)
  MAXSEQS     max_num_seqs (default 1)
  CTX         max_model_len (default 8192)
  UTIL        gpu_memory_utilization (default 0.92)
  DEPTH       synthetic prompt length in tokens (default: short fixed prompt)
  NSEQ        parallel sequences in the timed run (default 1)
  COLD=1      skip the warmup so the timed run pays cold prefill (TTFT probe)
  MAXTOK      timed generation length (default 256)
"""
import os
import time

MODEL = "models/local/Qwen3.8-Flash-Next-W4A16-INT4PLE"  # deploy-repo-relative
SHORT_PROMPT = "Write a Python function that reverses a linked list."

EAGER = os.environ.get("EAGER", "0") == "1"
MTPK = int(os.environ.get("MTPK", "3"))
CAPTURES = os.environ.get("CAPTURES", "").strip()
BATCHTOK = int(os.environ.get("BATCHTOK", "2048"))
MAXSEQS = int(os.environ.get("MAXSEQS", "1"))
CTX = int(os.environ.get("CTX", "8192"))
UTIL = float(os.environ.get("UTIL", "0.92"))
DEPTH = int(os.environ.get("DEPTH", "0"))
NSEQ = int(os.environ.get("NSEQ", "1"))
COLD = os.environ.get("COLD", "0") == "1"
KVB = float(os.environ.get("KVB", "0"))  # GiB; 0 = profiler-derived
MAXTOK = int(os.environ.get("MAXTOK", "256"))


def build_prompt(depth: int) -> str:
    if depth <= 0:
        return SHORT_PROMPT
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    para = (
        "The deployment layer keeps compose files, exporters and operational "
        "scripts in one repository per artifact type. Validation is a config "
        "check plus a pytest suite, and deploy verification happens in the "
        "live environment. "
    )
    ids = tok.encode(para)
    reps = depth // len(ids) + 1
    text = tok.decode(ids * reps)[: depth * 6]
    n = len(tok.encode(text))
    while n < depth:
        text += para
        n = len(tok.encode(text))
    ids = tok.encode(text)
    text = tok.decode(ids[:depth])
    print(f"DEPTH_PROMPT tokens={len(tok.encode(text))}")
    return text + "\n\nSummarize the passage in one sentence."


def main() -> None:
    from vllm import LLM, SamplingParams

    comp = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    if CAPTURES:
        comp["cudagraph_capture_sizes"] = [int(x) for x in CAPTURES.split(",")]
    elif MTPK:
        comp["cudagraph_capture_sizes"] = [1, MAXSEQS * (MTPK + 1)]
    kwargs = {"compilation_config": comp}
    if MTPK:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": MTPK,
            "draft_sample_method": "probabilistic",
        }
    prompt = build_prompt(DEPTH)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=CTX,
        max_num_seqs=MAXSEQS,
        max_num_batched_tokens=BATCHTOK,
        gpu_memory_utilization=UTIL,
        **({"kv_cache_memory_bytes": int(KVB * 2**30)} if KVB else {}),
        enforce_eager=EAGER,
        **kwargs,
    )
    sp = SamplingParams(max_tokens=MAXTOK, temperature=0, ignore_eos=True)
    if not COLD:
        llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0, ignore_eos=True), use_tqdm=False)
    t0 = time.perf_counter()
    out = llm.generate([prompt] * NSEQ, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in out)
    label = (f"eager={int(EAGER)} mtp={MTPK} batch={BATCHTOK} seqs={MAXSEQS} "
             f"ctx={CTX} util={UTIL:.2f} kvb={KVB:g} depth={DEPTH} nseq={NSEQ} cold={int(COLD)}")
    print(f"BENCH {label} tokens={n} wall={wall:.2f}s agg_rate={n / wall:.2f} tok/s")


if __name__ == "__main__":
    main()
