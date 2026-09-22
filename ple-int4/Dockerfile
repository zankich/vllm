# flashnext-int4ple serve image.
# Base MUST be the devel flavor: a runtime JIT step after weight load (MoE
# setup) invokes torch cpp_extension, which requires nvcc and /usr/local/cuda;
# runtime-only bases fail with "Could not find nvcc and default
# cuda_home='/usr/local/cuda' doesn't exist". ubuntu24.04's python3 is 3.12,
# satisfying the cp38-abi3 wheel; pip nvcc sdists do not build, so the toolkit
# comes from the base image. gcc + Python.h are present for Triton's JIT shim.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# DEPRECATED: this pins the stock 0.29.1 nightly wheel, so building it yields
# upstream vLLM without the fork patches (no misaligned-group exclusion, no
# integrity/pin layers). Build the fork image instead: the root Dockerfile via
# scripts/build-fork-image.sh, which applies forks/v0.30.0z.patch and installs
# this plugin. Kept only as a minimal reference for the plugin's runtime deps.
ARG VLLM_WHEEL=https://wheels.vllm.ai/ba2ae9f23961ac67bc5c055da8c26fbd660989c6/vllm-0.29.1rc1.dev102%2Bgba2ae9f23-cp38-abi3-manylinux_2_28_x86_64.whl

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages "${VLLM_WHEEL}"

COPY pyproject.toml /opt/ple-int4/pyproject.toml
COPY ple_int4/ /opt/ple-int4/ple_int4/
RUN pip3 install --no-cache-dir --break-system-packages /opt/ple-int4

# NCCL/pinned-table working set: default shm is too small for 4 TP workers.
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
WORKDIR /workspace
