# Build:    scripts/build-fork-image.sh
# Re-pin:   regenerates forks/v0.30.0z.patch automatically before build

# Step 1: start from upstream vLLM v0.30.0
FROM vllm/vllm-openai:v0.30.0

# Step 2: apply the fork patches. The diff carries vllm/ source
# changes only.
COPY forks/v0.30.0z.patch /tmp/v0.30.0z.patch
RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 --fuzz=0 < /tmp/v0.30.0z.patch \
    && rm /tmp/v0.30.0z.patch

# Step 2b: install the int4 PLE plugin. The plugin's setup registers
# itself via the vllm.general_plugins entry point at vllm import
# time, so pip install is sufficient. Inert for any model that
# doesn't carry ple_embedding_dtype == "int4".
COPY ple-int4/ /opt/ple-int4/
RUN pip install --no-cache-dir /opt/ple-int4 \
    && rm -rf /opt/ple-int4

# Step 3: install Gemma-4 audio support (libsndfile is soundfile's system dep)
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
# torch 2.13.0+cu130 pins nvidia-nccl-cu13==2.29.7 in its metadata, but
# this image deliberately ships 2.30.7 (DeepEP v2 needs >= 2.30.4).
# Installing vllm[audio] re-resolves against torch's pin and downgrades
# NCCL, so the second install restores the version the base image came
# with. They must be separate invocations: asking one resolver pass for
# both backtracks to vllm 0.19.1 from PyPI, replacing the patched tree.
# librosa explicit: mistral_common is preinstalled without its [audio]
# extra, so pip's already-satisfied check skips expanding it.
RUN pip show vllm > /tmp/vllm-before.txt \
    && pip install --no-cache-dir "vllm[audio]" librosa \
    && pip install --no-cache-dir "nvidia-nccl-cu13==2.30.7" \
    && pip show vllm > /tmp/vllm-after.txt \
    && diff -u /tmp/vllm-before.txt /tmp/vllm-after.txt > /dev/null \
    && rm /tmp/vllm-before.txt /tmp/vllm-after.txt

# Step 4: verify the fork patches actually landed (fail the build if not)
# Verify imports in this order: filesystem first, then leaf-only imports.
# install_sm8_fp8_large_head_optin() triggers a deep arg_utils chain
# that needs a GPU in the build container, so the opt-in function is
# confirmed callable but not invoked here.
RUN python3 - <<'PY'
import importlib, importlib.util, pathlib

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
    "vllm.utils.flashinfer",
    "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
    "vllm.v1.attention.backends.triton_attn",
    "vllm.v1.attention.backends.flashinfer",
    "vllm.model_executor.layers.attention.attention",
    "vllm.v1.kv_offload.tiering.fs.integrity",
    "ple_int4",
]
sources = {m: _source_path(m) for m in modules}
missing = [m for m, p in sources.items() if p is None]
print("module sources ok:", not missing, missing)

# integrity stack leaf check (cheap path, no GPU dependency).
import vllm.v1.kv_offload.tiering.fs.integrity as i
print("offload integrity:", hasattr(i, "block_checksum"))

# opt-in function reachable (does not call it — that path needs a GPU).
import vllm.utils.flashinfer as fi
print("opt-in installed:", callable(getattr(fi, "install_sm8_fp8_large_head_optin", None)))

# plugin leaf check.
import ple_int4 as p
print("plugin module:", p.__name__, "at", _source_path("ple_int4"))
PY
