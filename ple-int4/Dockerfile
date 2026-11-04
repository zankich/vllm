# int4 PLE serve image.
# Base MUST be the devel flavor: a runtime JIT step after weight load (MoE
# setup) invokes torch cpp_extension, which requires nvcc and /usr/local/cuda;
# runtime-only bases fail with "Could not find nvcc and default
# cuda_home='/usr/local/cuda' doesn't exist". ubuntu24.04's python3 is 3.12,
# satisfying the abi3 wheel; pip nvcc sdists do not build, so the toolkit
# comes from the base image. gcc + Python.h are present for Triton's JIT shim.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# DEPRECATED: this installs the stock PyPI vllm wheel, so building it
# produces upstream vLLM without the fork patches. ple_int4.install()'s
# pins hash-match the stock modules and cannot catch that, so the image
# serves silently unpatched. Build the fork image instead: the root
# Dockerfile applies forks/v0.33.0z.patch and installs this plugin.
# Kept only as a minimal reference for the plugin's runtime deps.

ARG VLLM_VERSION=0.33.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages "vllm==${VLLM_VERSION}"

COPY pyproject.toml /opt/ple-int4/pyproject.toml
COPY ple_int4/ /opt/ple-int4/ple_int4/
RUN pip3 install --no-cache-dir --break-system-packages /opt/ple-int4

# NCCL/pinned-table working set: default shm is too small for 4 TP workers.
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
WORKDIR /workspace
