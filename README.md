<!-- markdownlint-disable MD001 MD041 -->
<!-- fork-preamble-start -->
# zankich/vllm — fork of vllm-project/vllm

Fork of vLLM carrying model-specific enablement and fixes (Qwen3.8-27B,
Qwen3.8-Flash-Next, Gemma-4) plus cross-model patches for KV-offload
integrity and restore correctness, FP8 KV serving on SM80-class GPUs,
and fixes to the upstream Anthropic `/v1/messages` endpoint. Upstream
vLLM is excellent; this fork exists to carry fixes that had not shipped
in a release at deploy time. See what this fork changes with:

```bash
git log v0.33.0..HEAD --oneline        # everything on top of the tag
git log v0.33.0..HEAD --stat          # full delta of the upstream tag
```

Pre-built images: [zankich/vllm-openai on Docker Hub](https://hub.docker.com/r/zankich/vllm-openai/tags).
Each `vX.Y.Zz` tag is built from the matching branch tip with the root
`Dockerfile`; `latest` tracks the newest `z` tag.

## Branches

- `v0.33.0z` — current, on the v0.33.0 tag; carries the cross-model offload-integrity, anchor and all-reduce patches, the Qwen3.8 enablement, and the Anthropic `/v1/messages` fixes, with synchronous small-hit loading sized through `sync_load_budget_bytes` in the deploy config, a vanished load reported as a failed receive under `kv_load_failure_policy`, and the offload region keeping its registration lock and orphan reclaim. Every suite in the pinned gate ran against the stock v0.33.0 baseline with no new failures at the branch tip, the fork-only tests accounted. Verified live: sm_86 FP8 KV boot with the stock warmup and cudagraph profile, the cross-rank registration-lock boot at TP2, a hybrid cold-restart restore through the fs tier (restores exact at the chunk grid with partial tails recomputed, a hard-killed engine's region reclaimed at the next boot), and FP8 QSA decode numerics identical to the stock reference band. Serving validation is discharged: a validation instance derived from the production deployment's resolved configuration served the production model through a full workload, with decode and speculative-acceptance parity with the stack it replaces.
- `v0.32.0z` — previous release branch, on the v0.32.0 tag

## Cross-model patches

Apply to every model this fork serves.

| commit | what it does | origin |
| --- | --- | --- |
| `Fix intermittent offload-region pinning failure` | concurrent `cudaHostRegister` of the shared region across TP ranks intermittently fails and poisons the CUDA context; registration, through upstream's `host_register` helper from [#51956](https://github.com/vllm-project/vllm/pull/51956), is now serialized by an exclusive flock on a dedicated `.reglock` file beside the region, held across the whole chunked registration whichever range set the rank registers, so it can never conflict with the shared liveness marks every region participant holds on the region itself, and upstream's chunked registration from [#51081](https://github.com/vllm-project/vllm/pull/51081) drains and rolls back a failed chunk with the region left pageable under a warning. A rank that takes the lock re-verifies that the locked file is still the one the path names and retries otherwise, so no removal of the `.reglock` file by this tree can split the critical section; the lock file's fd is closed whenever the re-check raises, and a symlinked reglock path is refused with `O_NOFOLLOW` since `/dev/shm` is world-writable; region cleanup removes the lock through the same guarded removal, so a graceful shutdown leaves no residue per boot | fork-local |
| `Reclaim orphaned offload regions` | a SIGKILL'd engine leaks `/dev/shm/vllm_offload_*.mmap`, wedging the next boot on shared `/dev/shm`; sweep at construction reclaims regions whose exclusive flock can be taken. The region is split across shm files beneath one contiguous mapping and only the base file carries the engine's liveness mark, so the base is the probe for the whole region: once it proves orphaned, the numbered segment files are unlinked with it. A zero-length base is a region mid-creation — the creator sizes the base last — and is skipped inside an age window, then falls through to the lock probe once it can belong to no live creation, which reclaims the whole-gigabyte segment residue a hard kill in the sizing window leaves behind; the sweep covers the booting engine's own name too, so a prior incarnation that died mid-creation under the same engine id cannot strand its files or wedge the reboot on the joiner's size wait. After the lock probe the sweep re-verifies that its handle still names the path before unlinking, and the segment unlinks stop if the base name reappears, so a creator that wins the name back mid-sweep keeps its region. A `.reglock` whose region name is already gone is swept by age so a booting engine's lock is never taken for residue | [upstream PR #54124](https://github.com/vllm-project/vllm/pull/54124), adapted |
| `Bind fs-tier blocks to their keys, detect replaced storage` | the tier's files were content-addressed by path only: full-length wrong-for-key bytes restored silently (a real incident class — index/path confusion, replaced storage, pruner-damaged trees). Each store records a key-bound checksum in a `user.vllm_kv_integrity` xattr on the payload, and loads verify before any block is trusted; a missing or foreign record, a storage-identity change under a live tier, or a post-read mismatch rejects the load, removes the payload, and degrades to a miss and a cold recompute — never a partial trust of the rest of the batch. Construction probes xattr support and fails loud rather than serving as a silent 100% miss | fork-local, Linux-only |
| `fs tier: stop re-minting integrity records over skipped payloads` | the batch store skips writing payloads whose files already exist, but the record mint ran unconditionally from the live view — a cascaded re-store of a key whose recompute quantized differently (FP8 KV) bound the record to bytes that never reached the file, and every later load failed the checksum deterministically, was removed, and the next cascade re-poisoned the fresh file. The store now mints only for files absent before the batch write; definite-corruption removals log a distinct `integrity-<reason>` line and emit `OffloadingEvent(removed=True)` so the pruner feed learns the key died; a post-check mismatch carries `num_succeeded` so the verified prefix keeps its verdict | fork-local |
| `CPU shm tier: in-memory slot checksums` | post-store slot clobber, aliasing, and torn writers had no detection anywhere in the cascade. `complete_store` records `sha256(key, slot bytes)`, lookup re-verifies once per key per request, and a mismatch answers MISS, evicts the corrupt block, and emits a removal event; chunks referenced by an in-flight load skip the verify so the copy already moving completes, and the next request's first lookup rejects. `verify_stored_slot` exposes the record as the cascade cross-check source for secondary tiers. `prepare_load` pre-scans for a key this step's lookup confirmed that is no longer in the cache and raises `OffloadKeyVanishedError` before consuming any key, so earlier keys keep their ref count and the load stays all-or-nothing. In-memory carrier — the CPU tier has no cross-restart reuse, regions die with their engine | fork-local |
| `offload cascade: store-time cross-check against the armed primary tier` | store-side self-verification closing the blind spot the load-side integrity could never see: it binds a key to the bytes AS STORED, so a mislabel at write time (foreign KV persisted under a correct key) verifies clean forever. The tiering manager arms the CPU primary tier's slot checksums and wires `verify_stored_slot` into every secondary tier that declares the cross-check slot; the fs store task hashes the bytes it is about to persist and refuses the write on divergence — a refusal costs a miss and recompute, a silent write costs permanent poison. `None` (no record, checksums off) stays inert so the load-side checks remain the only guard | fork-local |
| `offloading: anchor the partial-tail lookup on the full-attention prefix` | a recurrent align group keeps one state per producer request at that prompt's tail, so requiring every group to hit at the complete-chunk grid hid every partial tail beyond the first recurrent block and no cross-restart restore could ever assemble on a hybrid model — measured live: zero restore bytes across four eval shapes while stores flowed and the full-attention group hit alone. The lookup now anchors the search on the full-attention complete chunks and checks the remaining groups only at the candidate boundary, where their state actually lives; EAGLE/MTP draft groups opt out. The anchor only raises the boundary walk's floor — shapes the stock walk already serves keep their result — and the anchor chunk ceiling floors under either token cap, so the anchor never probes past a budget the walk would then reject wholesale | fork-local minimal anchor; matches the open upstream bug [vllm-project/vllm#53569](https://github.com/vllm-project/vllm/issues/53569), approach after [vllm-project/vllm#58618](https://github.com/vllm-project/vllm/pull/58618) |
| `OffloadingConnector: per-request restore-accounting line` | one INFO line per restore with external hits carrying the assembly arithmetic (`prompt`/`local`/`ext`/`boundary`/`keys`/`chunk`) plus ERROR violations on boundary-exceeds-request — the bound is the request's total token length, not the prompt, so a resumed request whose stored chunks cover its sampled output tokens is not flagged — and on non-window capacity below the restored extent; the correlation instrument for restore-shape debugging, and the restore observation point for synchronous loads | fork-local |
| `deploy config: synchronous small-hit loading sized by budget` | hits within the per-worker `sync_load_budget_bytes` load synchronously in the scheduling step; the rest defer to the async path. The budget is a deploy-config value sized from the measured restore hit-size histogram (per-request hit tokens times the per-token KV bytes of the served model, times the per-step concurrent-hit count), carried in the deployment repository as a labeled change | upstream mechanism, labeled deploy-config change |
| `offloading: report a vanished load as a failed receive` | a key a request's lookup confirmed can be gone by load time — the deferrable async load parks the request and the key is evicted across the defer window by a corruption rejection or idle-chunk eviction — and the manager raises for it at `prepare_load`. Within one scheduler step this cannot happen: the manager lock spans lookup through `prepare_load`, so a synchronous load never observes a key vanishing between the two. The scheduler catches the error, clears the request's load ranges, records it in the batch's failed set, and every rank reports the request in both `finished_recving` and `failed_recving` so the aggregator's intersection count completes and the failure surfaces. `kv_load_failure_policy` then decides the request's fate: `"fail"` finishes it with an error, `"recompute"` resets it to a local recompute where the vanished key answers MISS | fork-local |
| `Surface per-request spec-decode metrics on the Anthropic messages API` | `--per-request-spec-decode-metrics` stats reach `/v1/chat/completions` upstream but were dropped by the `/v1/messages` converter; both the response and the final `message_delta` stream event now carry the same `metrics.speculative_decoding` field, serialized only when populated so the flag-off wire carries no extension key | fork-local |
| `anthropic streaming: split usage on every chunk, map engine error codes` | `message_start` carried `input_tokens` = the whole prompt because the OpenAI generator attached `prompt_tokens_details` only to the final usage chunk even though the cache counts are read at first_iteration, while `message_delta` carried the split — at `prompt == cached + created` a field-wise client merge kept the whole prompt next to cache fields that also sum to it, so a merging client counted 2x the prompt and compacted at half the true context. The generator now stamps the details on every usage-carrying chunk, only when non-None so flag-off wire output stays byte-identical; both events agree by construction and nothing is buffered. Engine error codes map through upstream's [#62294](https://github.com/vllm-project/vllm/pull/62294) vocabulary table, with the single addition of 529 → `overloaded_error` so retrying clients recognize Anthropic's own overload status | fork-local |
| `custom all-reduce: P2P-aware fully-connected probe, expert-parallel group` | two coordinated changes: `NvmlCudaPlatform.is_fully_connected` accepts generic P2P read/write when NVLink is absent (stock requires NVLink specifically, rejecting PCIe-only boxes whose every pair has working P2P from the one-shot IPC path at world_size > 2), and `group_allows_custom_allreduce()` lets the `ep` group build CustomAllreduce alongside `tp` ([#54371](https://github.com/vllm-project/vllm/pull/54371)'s ETP-scoped prefix gate excluded it as collateral; decode-time MoE combine all-reduces then pay the NCCL latency floor). The dispatch chain still falls through to PYNCCL on size/dtype gates, so large combines keep the ring path. The vendored NVML P2P caps READ constant carries a trailing comma that made it a tuple, which the widened probe's first ctypes query would reject — the fix rides the widening. Mutually exclusive with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (workers die at custom_all_reduce.cuh:164) | fork-local |

## Qwen3.8 patches

### 27B stack

| commit | what it does | origin |
| --- | --- | --- |
| `Enforce thinking-budget wrap-up sentence` | prepends a pre-tokenized wrap-up sentence to the forced `</think>` close at thinking-budget exhaustion, with the budget reduced by the wrap-up length at init so the phrase fits inside the original allocation. Under spec decode the wrap-up walk advances only on tokens that actually landed (never on drafted tokens the rejection sampler may discard), and the force window pins the first diverging verify row plus the bonus row — pinning everything was measured worse than pinning nothing. Anthropic `/v1/messages` `thinking.budget_tokens` maps to `thinking_token_budget`. dormant unless `VLLM_THINKING_WRAPUP_TOKEN_IDS` is set | fork-local |

### Flash-Next

Upstream v0.33.0 serves FP8 E4M3 KV for the QSA family on SM8x directly:
the below-SM89 byte decode rides a uint8 cache view with folded scales
inside upstream's kernel, and the warmup is capability-aware, so no
software-decode or unfused-write patch is needed on this base. What
this fork adds for Flash-Next is calibrated static KV scales and the
int4 PLE layer for memory-constrained Ampere hosts from this repo's
`ple-int4/` (`vllm.general_plugins` entry point; on a non-int4 config
set `VLLM_PLUGINS=""` to unbind it and get truly stock behavior).

| commit | what it does | origin |
| --- | --- | --- |
| `QSA: static KV scale sidecar loader` | `VLLM_QSA_KV_SCALES` applies calibrated scalar K/V dequant scales from a JSON sidecar after checkpoint weights are placed and before profiling or cudagraph capture, filling both the `_k_scale`/`_v_scale` tensors the cache writer quantizes with and the float mirrors the sparse-attention kernel folds into its scales, so a sidecar that fills only one form cannot decode at scale 1.0 against a calibrated writer. Fail-closed: partial coverage under strict mode, non-scalar, non-positive or non-finite values, and non-fp8 layers all raise; sidecar names absent from the local module are skipped under pipeline parallelism (they belong to another rank's stage) while strict coverage of local layers still holds | after [halt95/qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s) reader design (Apache-2.0) |
| `QSA: KV clip counter and absmax calibration tooling` | `VLLM_QSA_KV_COLLECT` records a GPU-side running absmax of the pre-quantization BF16 K/V per layer, persisting per-rank JSON for an offline fail-closed merge into the sidecar (`scale = absmax × margin / 448`). `VLLM_QSA_KV_CLIP_COUNT` counts K/V elements that would saturate E4M3 after scaling — the under-coverage guard for a calibrated sidecar — with per-layer totals drained once at engine shutdown, and `VLLM_QSA_KV_CLIP_READER` adds a live reader restricted to diagnostic `--enforce-eager` boots: CUDA forbids sync reads, pinned-host allocation, stream creation and event queries from another thread during graph capture, and no API exposes capture state to that thread. Both readers read the BF16 K/V that only the unfused writer materializes, so `use_fused_qsa_prepare` gains a calibration switch that forces the unfused writer while either is armed | after halt95's collector/counter design, margin 1.10 rationale from their clip-count measurements |
| the `ple-int4/` plugin | int4 PLE for Flash-Next on memory-constrained Ampere, riding upstream's `lookup_dtype`/`lookup_from_pinned` seam: no `_lookup` override, no eager bf16 buffer, the lazy prefetch buffer allocates at the activation dtype. Unsharded int4 tables are refused loudly at weight load rather than loading wrong bytes silently. `ple_int4.install()` pins the stock module hashes it patches and refuses to serve on drift | fork-local |

## Gemma-4 patches

Gemma-4 (31B-it, W8A16 + FP8-KV checkpoints) has a 256-dim QK head and
512-dim value head — FlashInfer's large-head class. The shipped
FlashInfer build's precompiled sm80 cache covers e4m3 at head_dim 256
only, so the model's 512-dim value head under fp8 KV has no FlashInfer
kernel on SM8x and fp8-KV serving rides the Triton path; text-only
FlashInfer serving needs a bf16 KV cache there.

| commit | what it does | origin |
| --- | --- | --- |
| `triton attention: serve FP8 KV on SM80+ via e5m2 storage` | stock gates the Triton FP8 KV path on SM89+ because triton cannot compile fp8e4nv below SM89. On SM80/SM86, quantized KV instead serves as e5m2 storage, which triton compiles natively there: the backend gate drops to SM80, the impl resolves its fp8 view dtype to e5m2 below SM89, the cache-store wrappers quantize K/V in torch (round-to-nearest-even, since triton's implicit e5m2 cast rounds ties away from even) and view the caches as e5m2 so the kernel's is-fp8 input branch stores raw bytes, and decode dequantizes e5m2 tiles in software. Attention layers skip QuantFP8 query setup whenever the effective storage dtype is e5m2 — both the explicit `fp8_e5m2` config and the generic `fp8` string below SM89 resolve there, so both are servable flavors on SM80/86, with queries staying bf16 for in-kernel KV dequant; an e4m3 query against e5m2 storage would leave the query-descale guard without a scale while the kernel still compiles the scale load, so the launcher now rejects that combination with a clear error before launch. Scales calibrated for e4m3 semantics carry no range risk in e5m2 storage. Sub-SM90 fp8-KV launches pin num_stages to 1 and halve the prefill tile for heads above 256, keeping head_dim-512 tiles inside the SM80/86 shared-memory limit | fork-local |

## Rebase policy

Each upstream release: check which patches upstream has absorbed
(`git merge-base --is-ancestor <upstream-sha> <tag>`), re-port the rest.
Patches here exist to be deleted — the permanent fixes are the
fork-local ones until upstream takes them.

<!-- fork-preamble-end -->

<!-- markdownlint-disable MD001 MD041 -->
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
vllm download-kernels  # precompiled kernels for faster startup
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
