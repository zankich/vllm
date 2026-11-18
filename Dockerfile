# Build:    mkdir -p forks \
#             && git diff v0.34.0 v0.34.0z -- vllm/ > forks/v0.34.0z.patch \
#             && docker build -t 0.34.0z-rc -f Dockerfile .

# Step 1: start from upstream vLLM v0.34.0
FROM vllm/vllm-openai:v0.34.0

# Step 2: apply the fork patches. The diff carries vllm/ source
# changes only.
COPY forks/v0.34.0z.patch /tmp/v0.34.0z.patch
RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 --fuzz=0 < /tmp/v0.34.0z.patch \
    && rm /tmp/v0.34.0z.patch

# Step 3: audio serving deps; the base image already ships
# libsndfile1, soundfile's system dep. torch 2.14.0 pins
# nvidia-nccl-cu13==2.30.7 in its metadata, but this image
# deliberately ships 2.31.2, the version upstream's NCCL_VERSION
# build arg installs into the base image. Installing vllm[audio]
# re-resolves against torch's pin and downgrades NCCL, so the second
# install restores the version the base image came with. They must
# be separate invocations: asking one resolver pass for both
# backtracks to a PyPI vllm release, replacing the patched tree.
# librosa explicit: mistral_common is preinstalled without its [audio]
# extra, so pip's already-satisfied check skips expanding it.
RUN pip show vllm > /tmp/vllm-before.txt \
    && pip install --no-cache-dir "vllm[audio]" librosa \
    && pip install --no-cache-dir "nvidia-nccl-cu13==2.31.2" \
    && pip show vllm > /tmp/vllm-after.txt \
    && diff -u /tmp/vllm-before.txt /tmp/vllm-after.txt > /dev/null \
    && rm /tmp/vllm-before.txt /tmp/vllm-after.txt

# Step 3.5: stamp the fork version. The base image's build
# generates vllm/_version.py from the upstream tag via
# vcs-versioning, so the image reports "0.34.0"; the fork layer
# must report the fork version. Only the version string changes:
# __version_tuple__ and commit_id stay upstream's (the tuple feeds
# collect_env's per-element checks), and pip's dist metadata stays
# 0.34.0 deliberately -- the Step 3 resolver pass treats the
# installed dist as its input, and a version string that exists
# nowhere on the index is a resolver-backtrack trap. Assert the
# base value first so a base-tag drift fails the build instead of
# silently stamping over something unexpected.
RUN python3 - <<'PY'
import pathlib

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/_version.py"
)
src = p.read_text(encoding="utf-8")
old = "__version__ = version = '0.34.0'"
assert old in src, f"unexpected _version.py: {src!r}"
src = src.replace(old, "__version__ = version = '0.34.0z'")
p.write_text(src, encoding="utf-8")
print("stamped: vllm.__version__ = 0.34.0z")
PY

# Step 4: verify the fork patches actually landed (fail the build if not).
# Verify imports in this order: filesystem first, then leaf-only imports.
# Nothing here may need a GPU in the build container.
RUN python3 - <<'PY'
import importlib.util
import pathlib

def _source_path(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin in (None, "namespace"):
        # namespace package: any __init__.py counts, but for a vendor
        # overlay we expect a real one. Check the dir directly.
        paths = spec.submodule_search_locations or []
        return pathlib.Path(paths[0]).resolve() if paths else None
    return pathlib.Path(spec.origin).resolve()

modules = [
    "vllm.model_executor.layers.attention.attention",
    "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
    "vllm.v1.attention.ops.triton_unified_attention",
    "vllm.v1.attention.backends.triton_attn",
    "vllm.models.qwen4_exp.nvidia.qsa",
    "vllm.v1.kv_offload.tiering.fs.integrity",
    "vllm.v1.kv_offload.cpu.shared_offload_region",
    "vllm.distributed.device_communicators.cuda_communicator",
]
sources = {m: _source_path(m) for m in modules}
missing = [m for m, p in sources.items() if p is None]
print("module sources ok:", not missing, missing)
if missing:
    raise SystemExit(1)

# offload integrity stack leaf check (cheap path, no GPU dependency).
import vllm.v1.kv_offload.tiering.fs.integrity as integrity
print("offload integrity:", callable(integrity.block_checksum))

# QSA static KV scale sidecar loader. model imports qsa, so import
# it first: qsa imports model at module scope and is not importable
# ahead of it.
import vllm.models.qwen4_exp.nvidia.model as qwen_model
import vllm.models.qwen4_exp.nvidia.qsa as qsa
print("qwen4_exp nvidia model:", qwen_model.__name__)
print("qsa sidecar loader:", callable(qsa.load_qsa_static_kv_scales))

# offload region registration lock.
import vllm.v1.kv_offload.cpu.shared_offload_region as region
print("offload reglock:", getattr(region, "_REGLOCK_SUFFIX", None) == ".reglock")

# vanished-load failure symbol the connector catches.
import vllm.v1.kv_offload.base as offload_base
print("vanished-key error:", hasattr(offload_base, "OffloadKeyVanishedError"))

# flashinfer must stay at the version the SM8 large-head fp8
# shared-memory ruling is recorded against; a change re-opens the
# re-measurement the pin is recorded for.
import importlib.metadata as md
fi_version = md.version("flashinfer-python")
print("flashinfer-python:", fi_version)
assert fi_version == "0.7.3", (
    f"flashinfer-python {fi_version} != 0.7.3: the SM8 large-head "
    "fp8 shared-memory ruling is recorded against 0.7.3 and must "
    "be re-measured if this drifts"
)

# fork version stamp: the image must report the fork version, not
# the base release's.
import vllm
print("fork version:", vllm.__version__)
assert vllm.__version__ == "0.34.0z", (
    f"version stamp lost: {vllm.__version__}"
)
PY
