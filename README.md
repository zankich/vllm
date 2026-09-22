<!-- markdownlint-disable MD001 MD041 -->
<!-- fork-preamble-start -->
# zankich/vllm — fork of vllm-project/vllm

Fork of vLLM carrying model-specific enablement and
fixes (Qwen3.8-27B, Qwen3.8-Flash-Next, Gemma-4) plus cross-model
patches for KV-offload correctness, FlashInfer on SM8x, and the
Anthropic `/v1/messages` endpoint. Upstream vLLM is excellent;
this fork exists to carry fixes that had not shipped in a release at
deploy time. See what this fork changes with:

```bash
git log v0.30.0..HEAD --oneline        # everything on top of the tag
git log v0.30.0..HEAD --stat          # full delta of the upstream tag
```

## Branches

- `v0.30.0z` (default) — current, on the v0.30.0 tag; serves the Qwen3.8-27B stack (TP2, MTP, prefix caching, fp8 KV, tiered CPU + fs KV offload), Qwen3.8-Flash-Next (TP4+EP, MTP, UVA PLE; serving gates passed — int4 PLE, MTP and KV offload in one boot, restart-restore byte-compare PASS, benches within 6% of the reference nightly short-context), and Gemma-4 (TP2, MTP, fp8 KV, tiered offload)

## Cross-model patches

Apply to every model this fork serves.

| commit | what it does | origin |
|---|---|---|
| `Fix intermittent offload-region pinning failure` | concurrent `cudaHostRegister` of the shared offload region across TP ranks intermittently fails and poisons the CUDA context (warn-and-continue killed the next CUDA op); now flock-serialized across ranks, retried, and fails the boot loudly | fork-local, no upstream fix at port time |
| `Reclaim orphaned offload regions` | a SIGKILL'd engine leaks `/dev/shm/vllm_offload_*.mmap`, wedging the next boot on shared `/dev/shm`; sweep at construction reclaims regions whose exclusive flock can be taken (port of upstream [#54124](https://github.com/vllm-project/vllm/pull/54124), closed unmerged) | [upstream PR #54124](https://github.com/vllm-project/vllm/pull/54124), adapted |
| `Surface per-request spec-decode metrics on the Anthropic messages API` | `--per-request-spec-decode-metrics` stats reach `/v1/chat/completions` upstream but were dropped by the `/v1/messages` converter; both the response and the final `message_delta` stream event now carry the same `metrics.speculative_decoding` field | fork-local |
| `Bind fs-tier blocks to their keys, detect replaced storage` | the tier's files were content-addressed by path only: full-length wrong-for-key bytes restored silently (a real incident class — index/path confusion, replaced storage, pruner-damaged trees). Each store records a key-bound checksum in a `user.vllm_kv_integrity` xattr on the payload (sidecars in an earlier revision — same record bytes, xattr carrier halves the inodes and leaves nothing for the pruner to orphan); loads verify before and after the read, transient errors (ELOOP/EACCES/EIO) fail without deletion, and a payload whose storage identity changed under a live engine rejects the whole job to a cold recompute. Never partial trust. Construction probes xattr support and fails loud rather than silently missing | fork-local, Linux-only |
| `CPU shm tier: in-memory slot checksums` | post-store slot clobber, aliasing, and torn writers had no detection anywhere in the cascade. `complete_store` records `sha256(key, slot bytes)`; lookup re-verifies once per key per request and a mismatch answers MISS (nothing downstream can crash or misalign), evicts the corrupt block, and emits a removal event. In-memory carrier — the CPU tier has no cross-restart reuse, regions die with their engine | fork-local |
| `OffloadingConnector: per-request restore-accounting line` | one INFO line per restore with external hits carrying the assembly arithmetic (`prompt`/`local`/`ext`/`boundary`/`keys`/`chunk`) plus ERROR violations on boundary-exceeds-prompt and keys-cannot-cover — the correlation instrument for restore-shape debugging. The store-side companion was removed with the investigation it served: its invariants misfired on healthy traffic and its line fired per scheduled request per step | fork-local |
| `CPU tier: pin lookup-confirmed hits until load or finish` | store completions and their LRU evictions run on transfer threads asynchronously from the scheduler thread, so an unpinned lookup-confirmed hit could vanish between the connector's confirming lookup and prepare_load — a fatal `Block ... not found in cache` under restore-heavy load, and the mechanism behind repeated corrupt-output incidents (verified end to end under adversarial load before landing). A confirmed HIT now pins (ref_cnt-like, insertion-ordered per request); prepare_load's ref count takes the pin over, never-loaded pins release at request finish, and the corrupt-block reject path accounts for pins. Pressure against pinned keys surfaces as store refusal, never as key disappearance. Latent in stock | fork-local |
| `custom all-reduce: P2P-aware fully-connected probe, expert-parallel group` | two coordinated changes: `NvmlCudaPlatform.is_fully_connected` accepts generic P2P read/write when NVLink is absent (stock requires NVLink specifically, rejecting PCIe-only boxes whose every pair has working P2P from the one-shot IPC path at world_size > 2 — measured +4.5% decode at TP2 on this fleet), and `group_allows_custom_allreduce()` lets the `ep` group build CustomAllreduce alongside `tp` (#54371's ETP-scoped prefix gate excluded it as collateral; decode-time MoE combine all-reduces then pay the NCCL latency floor). The dispatch chain still falls through to PYNCCL on size/dtype gates, so large combines keep the ring path. Supersedes the ple-int4 plugin's TP4 force, which returned True unconditionally instead of probing. Mutually exclusive with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (workers die at custom_all_reduce.cuh:164) | fork-local |

## Qwen3.8 patches

### 27B stack

| commit | what it does | origin |
|---|---|---|
| `Enforce thinking-budget wrap-up sentence` | prepends a pre-tokenized wrap-up sentence to the forced `</think>` close at thinking-budget exhaustion, with a spec-decode-resync fix so multi-token wrap-ups survive MTP rejection sampling; Anthropic `/v1/messages` `thinking.budget_tokens` maps to `thinking_token_budget`. dormant unless `VLLM_THINKING_WRAPUP_TOKEN_IDS` is set | fork-local |

### Flash-Next

The in-tree PLE formats are BF16 and FP8 only, and the FP8 table pins
~48 GiB of host RAM — and its pinned-lookup kernel is fp8e4nv-native
Triton, which SM8x rejects at compile (no fallback; `_reduce_etp_`
all-reduces raw fp8 bytes on the same Hopper assumption), so on Ampere
the stock FP8 PLE is unservable outright. Both reasons point
memory-constrained Ampere hosts at the int4 PLE plugin from this
repo's `ple-int4/` (`vllm.general_plugins` entry point; on a non-int4
config set `VLLM_PLUGINS=""` to unbind it and get truly stock
behavior).

| commit | what it does | origin |
|---|---|---|
| `Exclude block-misaligned KV groups from offloading` | Flash-Next + MTP forms groups [800 x5, 8]: the QSA indexer `raw_key_cache`'s 8-token block cannot chunk-hash at the 800-token granularity, and the offloading path asserted at config build, then in scheduler key/load/store math once the group was dropped outright. Misaligned groups keep their positional entry with no layers — nothing registers, stores, loads, or lookups for them — while the main-model context offloads normally. The assert now carries upstream's `--enable-prefix-caching` suggestion, which does not apply to the indexer-block case | fork-local |

## Gemma-4 patches

Gemma-4 (31B-it, W8A16 + FP8-KV checkpoints) has a 256-dim QK head and
512-dim value head — FlashInfer's large-head class. Two serving modes
on SM8x: FlashInfer text-only (`--language-model-only`, e4m3 KV with
the calibrated scales), or Triton multimodal (image support, e5m2 KV —
FlashInfer does not support this model's multimodal attention). With
both opt-ins below, the full stack serves: MTP (the Gemma assistant
drafter), fp8 KV, and tiered CPU + fs KV offload on the same
cross-model integrity and pin patches above. Validated on v0.29.0z
under warm/churn/extend restore traffic with the restore-accounting
instrumentation reading full coverage; v0.30.0z has not booted Gemma-4
(weights absent) and awaits revalidation.

| commit | what it does | origin |
|---|---|---|
| `flashinfer: widen the SM8 large-head opt-in to fp8 one-byte KV` | FlashInfer gates all one-byte-KV large-head (head_dim > 256 on either QK or VO) FA2 modules to SM100+ and its only SM8 opt-in (`allow_nvfp4_sm8_large_head`) is not recognized for fp8, so Gemma-4's 512-dim value head under fp8 KV fails JIT on SM8x with "No supported CUDA architectures found". `install_sm8_fp8_large_head_optin()` in `vllm.utils.flashinfer` extends the opted-in prefill path to fp8_e4m3/e5m2; nvfp4 semantics and non-opted-in paths unchanged; the FlashInfer backend installs it at import and a layout drift raises instead of silently serving the gate | fork-local |
| `triton attention: serve FP8 KV on SM80+ via e5m2 storage` | FlashInfer does not support this model's multimodal attention and FLASH_ATTN rejects FP8 KV below SM90, so TRITON_ATTN is the only backend that serves Gemma-4 image support on pre-SM100 GPUs — and stock gates its FP8 KV path to SM89+ (triton cannot compile fp8e4nv below SM89). The gate widens to SM80+; below SM89 the quantized KV serves as e5m2 (selected by `kv_fp8_dtype_for_platform`), the cache-store wrappers quantize in torch with round-to-nearest-even (`quantize_kv_e5m2_sm80` — triton's implicit e5m2 cast rounds ties away from even), and large-head prefill tiles stage within SM80/86 shared memory (num_stages 1, halved tile for head_dim > 256). fp8_e5m2 with a calibrated kv_cache_scheme stays permitted below SM89 (e4m3-calibrated scales add mantissa noise but no range risk) and keeps raising on SM89+ | fork-local |

## Rebase policy

Each upstream release: check which patches upstream has absorbed
(`git merge-base --is-ancestor <upstream-sha> <tag>`), re-port the rest.
Patches here exist to be deleted — the
permanent fixes are the fork-local ones until upstream takes them.
The Flash-Next upstream picks (six cherry-picks, one graft, one cohort
sync) deleted wholesale on the v0.30.0 rebase: #54371, #54517, #54890,
#55272, #55375, #55535, #55513, and the cohort sync all landed upstream
before v0.30.0; the four absorbed cross-model cherry-picks (#52807,
#52771, #55712, #54288) are gone for the same reason. The
misaligned-group exclusion persists until upstream grows its own knob.

<!-- fork-preamble-end -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
