#!/usr/bin/env bash
# Serve for QSA K/V absmax calibration: the served deployment's environment
# and flags, plus VLLM_QSA_KV_COLLECT with --enforce-eager (the collector
# mutates state per step; graphs/compile would freeze or fold it), no
# speculative config (the draft head's QSA layer must not land in the dump),
# no prefix caching (every request must run its full prefill through the
# collector) and no KV offload connector.
#
# The venv must carry the ple_int4 plugin for the INT4PLE checkpoint:
#   uv pip install -e ple-int4
#
#   CKPT and CHAT_TEMPLATE point at your deployment's checkpoint dir and
#   chat template; COLLECT at an existing dump dir.
#   mkdir -p /tmp/qsa-calib-dumps
#   VENV=.venv CKPT=<ckpt> CHAT_TEMPLATE=<tpl> COLLECT=/tmp/qsa-calib-dumps bash calib/calib_launch.sh
#   .venv/bin/python calib/qsa_calib_traffic.py 8140 qwen3.8-flash-next   # another shell
#   .venv/bin/python calib/qsa_calib_merge.py /tmp/qsa-calib-dumps <out>.json --ranks 4 --layers 12 --margin 1.10
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:-${CKPT:?pass <checkpoint-dir> (or set CKPT)}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:?set CHAT_TEMPLATE=<chat-template.jinja>}"
VENV="${VENV:-$HERE/.venv}"
COLLECT="${COLLECT:?set COLLECT=<existing dump dir>}"
[ -d "$COLLECT" ] || { echo "COLLECT dir must exist: $COLLECT"; exit 1; }
[ -d "$CKPT" ] || { echo "checkpoint dir not found: $CKPT"; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# served deployment environment
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS HF_HUB_OFFLINE=1
export VLLM_QSA_KV_COLLECT="$COLLECT"
unset VLLM_QSA_KV_SCALES VLLM_QSA_KV_CLIP_COUNT PYTORCH_CUDA_ALLOC_CONF || true

exec "$VENV/bin/vllm" serve "$CKPT" \
  --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port "${PORT:-8140}" \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 2684354560 --kv-cache-dtype fp8_e4m3 \
  --quantization compressed-tensors \
  --limit-mm-per-prompt '{"image":1}' --mm-processor-cache-gb 0 \
  --enforce-eager --no-enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template "$CHAT_TEMPLATE"
