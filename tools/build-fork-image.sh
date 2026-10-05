#!/usr/bin/env bash
#
# usage:  tools/build-fork-image.sh [--no-cache] [tag]
# default tag: zankich/vllm-openai:0.31.0z
# the default build also stamps zankich/vllm-openai:latest so the manual
# push publishes both. an explicit tag argument never touches latest:
# experiment builds must not repoint what latest tracks.
#
# regenerates forks/v0.31.0z.patch from the current tree (so a freshly
# committed source change is always baked in), builds the Dockerfile at
# the repo root. does NOT push: that stays a manual step.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

NO_CACHE=""
if [[ "${1:-}" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
    shift
fi

DEFAULT_TAG="zankich/vllm-openai:0.31.0z"
TAG="${1:-$DEFAULT_TAG}"

# latest only on the canonical stable build, never on experiment tags
EXTRA_TAGS=()
if [[ "$TAG" == "$DEFAULT_TAG" ]]; then
    EXTRA_TAGS+=(-t "${TAG%%:*}:latest")
fi

# the diff is from vanilla vLLM (the upstream base) to the fork tip.
# the upstream base is the vLLM v0.31.0 image, NOT this branch.
# scope is -- vllm/ only: the ple-int4/ tree installs separately in
# the Dockerfile (pip install flattens the nested package layout
# to dist-packages/ple_int4/__init__.py, which is what the plugin's
# vllm.general_plugins entry point needs). repo-root infra
# (Dockerfile, tools/, .dockerignore) is not shipped. compares
# against the local v0.31.0z tip (both names point at the same
# commits; both the local branch and origin/v0.31.0z carry them
# once pushed).
mkdir -p forks
git diff v0.31.0 v0.31.0z -- vllm/ >forks/v0.31.0z.patch
echo "patch: forks/v0.31.0z.patch ($(wc -l <forks/v0.31.0z.patch) lines)"

docker build $NO_CACHE -t "$TAG" "${EXTRA_TAGS[@]}" -f Dockerfile "$REPO_ROOT"

echo "built: $TAG"
if [[ ${#EXTRA_TAGS[@]} -gt 0 ]]; then
    echo "built: ${TAG%%:*}:latest (same image)"
    echo "push manually: docker push $TAG && docker push ${TAG%%:*}:latest"
else
    echo "push manually: docker push $TAG"
fi
