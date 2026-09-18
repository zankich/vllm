<!-- markdownlint-disable MD001 MD041 -->
<!-- fork-preamble-start -->
# zankich/vllm — fork of vllm-project/vllm

Production fork for Qwen3.8 serving: the 27B mamba-hybrid stack (TP2, MTP
speculative decoding, prefix caching, fp8 KV, CPU + disk KV-offload
tiers across two serving instances, fs tier per instance) on
`v0.29.0-qwen`, and Flash-Next bring-up (TP4+EP, MTP, the PLE n-gram
table pinned host-side and read through UVA) on
`v0.29.0-qwen-flashnext`. Upstream vLLM is excellent;
this fork exists to carry fixes that had not shipped in a release at
deploy time. See what this fork changes with:

```bash
git log v0.29.0..HEAD --oneline        # everything on top of the tag
git log <upstream-tag>..HEAD --stat    # full delta of the last-upstream-tag
```

## Branches

- `v0.29.0-qwen` (default) — current, on the v0.29.0 tag
- `v0.29.0-qwen-flashnext` — `v0.29.0-qwen` plus the upstream PLE-UVA backport chain for Qwen3.8-Flash-Next (patch table below; serving gates pending)
- `v0.28.0-qwen` — previous generation, on the v0.28.0 tag

## Patch set on `v0.29.0-qwen`

| commit | what it does | origin |
|---|---|---|
| `Enforce thinking-budget wrap-up sentence` | prepends a pre-tokenized wrap-up sentence to the forced `</think>` close at thinking-budget exhaustion, with a spec-decode-resync fix so multi-token wrap-ups survive MTP rejection sampling; Anthropic `/v1/messages` `thinking.budget_tokens` maps to `thinking_token_budget`. dormant unless `VLLM_THINKING_WRAPUP_TOKEN_IDS` is set | fork-local |
| [`[Bugfix][KV Offload] ... truncate the load boundary` (#52807)](https://github.com/vllm-project/vllm/pull/52807) | mamba/recurrent groups legitimately hold unhashed blocks below the computed mark; the from-zero scan collapsed the load boundary and asserted (upstream [#50454](https://github.com/vllm-project/vllm/issues/50454)) | upstream, merged to main after the v0.29 branch cut |
| [`[Bugfix] ... stop zeroing offload hits under MTP/EAGLE` (#52771)](https://github.com/vllm-project/vllm/pull/52771) | with no annotated drafter group every group was treated as volatile-tail, zeroing the whole request's offload hit on shared-group MTP models | upstream, merged to main after the branch cut |
| [`[Bugfix][KV Offload] ... unaligned cache-hit boundaries` (#55712)](https://github.com/vllm-project/vllm/pull/55712) | SWA window coverage validation at unaligned hit boundaries | upstream, merged to main after the branch cut |
| `Fix intermittent offload-region pinning failure` | concurrent `cudaHostRegister` of the shared offload region across TP ranks intermittently fails and poisons the CUDA context (warn-and-continue killed the next CUDA op); now flock-serialized across ranks, retried, and fails the boot loudly | fork-local, no upstream fix at port time |
| `Reclaim orphaned offload regions` | a SIGKILL'd engine leaks `/dev/shm/vllm_offload_*.mmap`, wedging the next boot on shared `/dev/shm`; sweep at construction reclaims regions whose exclusive flock can be taken (port of upstream [#54124](https://github.com/vllm-project/vllm/pull/54124), closed unmerged) | [upstream PR #54124](https://github.com/vllm-project/vllm/pull/54124), adapted |
| `Surface per-request spec-decode metrics on the Anthropic messages API` | `--per-request-spec-decode-metrics` stats reach `/v1/chat/completions` upstream but were dropped by the `/v1/messages` converter; both the response and the final `message_delta` stream event now carry the same `metrics.speculative_decoding` field | fork-local |

## Patch set on `v0.29.0-qwen-flashnext`

`v0.29.0-qwen` plus the PLE-UVA backport: six upstream commits
cherry-picked from main and a qwen4_exp cohort sync from upstream
`c69d5d72a6` that composes #54517 with #54371's split, so
Qwen3.8-Flash-Next can serve
with MTP and KV offload on the fork base, where hybrid+MTP+offload is
proven on the 27B stack. Upstream's current line cannot boot this
model with the OffloadingConnector at all — hybrid block-size assert,
MTP+offload CUDA failure, CPU-tier shm size mismatch — which is why
#54371 was ported here rather than serving the nightly. The in-tree
PLE formats are BF16 and FP8 only, and the FP8 table pins ~48 GiB of
host RAM, so memory-constrained hosts need the int4 PLE plugin from
this repo's `ple-int4/` (`vllm.general_plugins` entry point).
above is inherited bit-identical: the six fork-patched files are 0-diff
against `v0.29.0-qwen`.

| commit | what it does | origin |
|---|---|---|
| [`[Bugfix][Qwen4Exp] ... state index strides in fused PLE conv` (#55375)](https://github.com/vllm-project/vllm/pull/55375) | fixes state index strides in the fused PLE convolution; brings in the new `nvidia/ops/ple.py` split module | upstream, cherry-picked from main |
| [`[Kernel] Remove unused fake implementation` (#55535)](https://github.com/vllm-project/vllm/pull/55535) | drops unused fake (meta) implementations across the ops wrappers, helion kernels, and qwen4_exp layers | upstream, cherry-picked from main |
| [`[Qwen3.8-Flash-Next] Remove torch.compile for NVIDIA implementation` (#55272)](https://github.com/vllm-project/vllm/pull/55272) | removes torch.compile from the NVIDIA model path; reshapes `model.py`/`ple_layer.py` to the state #54371's split applies against | upstream, cherry-picked from main |
| [`[Qwen3.8-Flash-Next] Support FP8 indexer cache for QSA` (#54890)](https://github.com/vllm-project/vllm/pull/54890) | FP8 cache for the QSA indexer; adds `nvidia/ops/qsa_indexer.py` | upstream, cherry-picked from main |
| [`Fix block FP8 MTP in ModelOpt mixed checkpoints` (#55513)](https://github.com/vllm-project/vllm/pull/55513) | routes block-FP8 routed experts to `Fp8MoEMethod` so FP8 MTP weights in ModelOpt mixed checkpoints load; hand-adapted, see below | upstream, cherry-picked from main, hand-adapted |
| [`[Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism` (#54371)](https://github.com/vllm-project/vllm/pull/54371) | the payload: the PLE n-gram table moves to `nvidia/ngram_embedding.py`, pinned host-side and read through UVA on a side stream, Engram tensor parallelism (ETP=TP); adds `vllm/config/engram.py` | upstream, cherry-picked from main |
| `Exclude block-misaligned KV groups from offloading` | Flash-Next + MTP forms groups [800 x5, 8]: the QSA indexer `raw_key_cache`'s 8-token block cannot chunk-hash at the 800-token granularity, and the offloading path asserted at config build, then in scheduler key/load/store math once the group was dropped outright. Misaligned groups keep their positional entry with no layers — nothing registers, stores, loads, or lookups for them — while the main-model context offloads normally. Verified by the restart-restore byte-compare protocol (garble PASS, ~3 GiB CPU-to-GPU restore) | fork-local |
| `Bind fs-tier blocks to their keys, detect replaced storage` | the tier's files were content-addressed by path only: full-length wrong-for-key bytes restored silently (a real incident class — index/path confusion, replaced storage, pruner-damaged trees). Each store records a key-bound checksum in a `user.vllm_kv_integrity` xattr on the payload (sidecars in an earlier revision — same record bytes, xattr carrier halves the inodes and leaves nothing for the pruner to orphan); loads verify before and after the read, transient errors (ELOOP/EACCES/EIO) fail without deletion, and a payload whose storage identity changed under a live engine rejects the whole job to a cold recompute. Never partial trust. Construction probes xattr support and fails loud rather than silently missing | fork-local, Linux-only |
| `CPU shm tier: in-memory slot checksums` | post-store slot clobber, aliasing, and torn writers had no detection anywhere in the cascade. `complete_store` records `sha256(key, slot bytes)`; lookup re-verifies once per key per request and a mismatch answers MISS (nothing downstream can crash or misalign), evicts the corrupt block, and emits a removal event. In-memory carrier — the CPU tier has no cross-restart reuse, regions die with their engine | fork-local |
| `Alignment shift for pure-tier flush-aligned restores` | the corrupt-output fingerprint: every corrupting restore was pure-tier (local == 0) and ended exactly on the full-attention chunk grid (14,400 = 9 x 1,600, three times) while the one clean control cut through a partial chunk (15,776). Opt-in via `kv_connector_extra_config {"alignment_shift": true}`: such restores return one FA chunk fewer hits so prefill recomputes that tail over the restored blocks — the overwrite is the protection, whichever side the defect is on. The load itself still covers the full confirmed window (a reduced boundary is not chunk-legal for coarser sibling groups; the first revision propagated the shaved count into the load geometry and crashed prepare_load under eviction churn on mixed-granularity configs). Fires only with no partial tail and no GPU-local hits | fork-local |
| [`[Qwen3.8-Flash-Next] Fuse Qwen4Exp PLE kernels` (#54517)](https://github.com/vllm-project/vllm/pull/54517) | load-critical for split-projection checkpoints: carries the `ple.key_proj`/`ple.value_proj` → merged `kv_proj` stacked-params remap, without which halt95-class checkpoints fail with `no module or parameter named layers.1.ple.key_proj`; also fuses the PLE kernels | upstream, composed via the cohort sync below |

Hand-adaptations forced by intermediate-commit drift (the deltas vs the
upstream commits as they landed on main):

- the qwen4_exp cohort (`vllm/models/qwen4_exp/`, `tests/models/qwen4_exp/`)
  is synced byte-identical to upstream `c69d5d72a6`, the tree the reference
  nightly served with, rather than hunk-surgery: #54517 (`f870b92976`)
  predates #54371's split, so picking it after the payload cannot compose
  whole-file (an earlier such pick, `af3e055369`, regressed the split and
  disconnected the ple-int4 plugin; superseded by the sync commit).
  `short_conv_attn.py` takes only #54517's own hunk, and
  `ngram_embedding.py` hashes to the ple-int4 plugin pin (`f3aaf292`);
- picked in true ancestry order (`28e605fb33` `199cb9b964` `d9105ea800`
  `94e26dd3dd` `60ad959b6f`), after which #54371 applied with no
  conflicts;
- conflicts in the qwen4_exp cohort resolved whole-file to the incoming
  side, because the v0.29.0-era context cannot merge hunk-wise: the
  `mutates_args` `output`→`residual_output` rename in `ple_layer.py`
  would otherwise pair a fork signature with an upstream registration;
- `modelopt.py` in #55513 keeps the fork's flat `QUANT_ALGOS`; main's
  `LINEAR_ALGOS` restructure, from an unpicked intermediate commit, was
  not taken. Only the fix is grafted: `_BLOCK_FP8_MOE_ALGOS`, the
  `Fp8Config` build with group-size validation, `has_blocked_weights`
  (which did not exist on this base), and the `Fp8MoEMethod`
  routed-experts branch;
- `nvidia/ops/qsa.py` and `ops/hc.py` ride the cohort sync to their
  upstream states; `vllm/config/engram.py` and `nvidia/ngram_embedding.py`
  are new files.

## Rebase policy

Each upstream release: check which patches upstream has absorbed
(`git merge-base --is-ancestor <upstream-sha> <tag>`), re-port the rest.
The commit messages record every hand-adaptation forced by
intermediate-commit drift; the flashnext chain's adaptations are the
bullets above, since its picks keep their upstream messages. Patches
here exist to be deleted — the
permanent fixes are the fork-local ones until upstream takes them.
The flashnext chain is six upstream cherry-picks, one graft, one cohort
sync, and one fork-local offloading fix (which persists until upstream
grows its own exclusion knob); the upstream part deletes wholesale at
the first final release the fork rebases onto that contains #54371 and
#54517 (both already in v0.29.1rc0).

---

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
