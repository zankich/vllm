# Build:    mkdir -p forks \
#             && git diff v0.33.0 v0.33.0z -- vllm/ > forks/v0.33.0z.patch \
#             && docker build -t 0.33.0z-rc -f Dockerfile .

# Step 1: start from upstream vLLM v0.33.0
FROM vllm/vllm-openai:v0.33.0

# Step 2: apply the fork patches. The diff carries vllm/ source
# changes only.
COPY forks/v0.33.0z.patch /tmp/v0.33.0z.patch
RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 --fuzz=0 < /tmp/v0.33.0z.patch \
    && rm /tmp/v0.33.0z.patch

# Step 2b: install the int4 PLE plugin. The plugin's setup registers
# itself via the vllm.general_plugins entry point at vllm import
# time, so pip install is sufficient. Inert for any model that
# doesn't carry ple_embedding_dtype == "int4".
COPY ple-int4/ /opt/ple-int4/
RUN pip install --no-cache-dir /opt/ple-int4 \
    && rm -rf /opt/ple-int4

# Step 3: audio serving deps; the base image already ships
# libsndfile1, soundfile's system dep. torch 2.14.0+cu130 pins
# nvidia-nccl-cu13==2.30.7 in its metadata, but this image
# deliberately ships 2.31.2. Installing vllm[audio] re-resolves
# against torch's pin and downgrades NCCL, so the second install
# restores the version the base image came with. They must be
# separate invocations: asking one resolver pass for both
# backtracks to vllm 0.19.1 from PyPI, replacing the patched tree.
# librosa explicit: mistral_common is preinstalled without its [audio]
# extra, so pip's already-satisfied check skips expanding it.
RUN pip show vllm > /tmp/vllm-before.txt \
    && pip install --no-cache-dir "vllm[audio]" librosa \
    && pip install --no-cache-dir "nvidia-nccl-cu13==2.31.2" \
    && pip show vllm > /tmp/vllm-after.txt \
    && diff -u /tmp/vllm-before.txt /tmp/vllm-after.txt > /dev/null \
    && rm /tmp/vllm-before.txt /tmp/vllm-after.txt

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
    "ple_int4",
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

# custom all-reduce EP-group gate.
import vllm.distributed.device_communicators.cuda_communicator as comm
print("custom all-reduce ep gate:", callable(comm.group_allows_custom_allreduce))

# thinking wrap-up env declared (unset is a no-op; presence is the check).
from vllm.envs import environment_variables
print(
    "thinking wrap-up env:",
    "VLLM_THINKING_WRAPUP_TOKEN_IDS" in environment_variables,
)

# flashinfer must stay at the version the SM8 large-head fp8 drop
# ruling was measured against; a change re-opens that ruling.
import importlib.metadata as md
fi_version = md.version("flashinfer-python")
print("flashinfer-python:", fi_version)
assert fi_version == "0.7.2.post1", (
    f"flashinfer-python {fi_version} != 0.7.2.post1: the SM8 large-head "
    "fp8 drop ruling was measured on 0.7.2.post1 and must be re-opened"
)

# plugin entry point registration: vllm.general_plugins names ple_int4.
from importlib.metadata import entry_points
plugin_eps = entry_points(group="vllm.general_plugins")
print("plugin registered:", any(ep.value == "ple_int4:install" for ep in plugin_eps))

# plugin pin check: install() exercises both pin hashes (nvidia + common)
# and raises on drift, failing the build at the pin check instead of at
# first serve boot on the host.
import ple_int4
print("plugin module:", ple_int4.__name__, "at", _source_path("ple_int4"))
try:
    ple_int4.install()
    print("ple_int4 pin check: ok")
except RuntimeError as e:
    print(f"ple_int4 pin check: FAILED — {e}")
    raise
PY
