"""G5b gate: full-checkpoint int4-PLE serve inside the capped container.

Prints the generation and top-5 logprobs of the first sampled token so a
degenerate distribution is visible immediately (the first attempt's `!!!!`
was caused by an index that loaded only the PLE files; the logprob spread is
how we tell 'model loaded' from 'logits collapsed' if anything recurs).
"""


def main() -> None:
    from vllm import LLM, SamplingParams

    MODEL = "models/local/Qwen3.8-Flash-Next-W4A16-INT4PLE"  # deploy-repo-relative

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
    )
    out = llm.generate(
        ["Write a Python function that reverses a linked list."],
        SamplingParams(max_tokens=64, temperature=0, logprobs=5),
    )
    text = out[0].outputs[0].text
    print("G5B_TEXT", repr(text))
    if out[0].outputs[0].logprobs:
        first = out[0].outputs[0].logprobs[0]
        for tid, lp in sorted(first.items(), key=lambda kv: -kv[1].logprob)[:5]:
            print(f"G5B_LP token={tid} logprob={lp.logprob:.3f}")
    print("G5B_OK")


if __name__ == "__main__":
    main()
