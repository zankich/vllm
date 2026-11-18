#!/usr/bin/env bash
# Simulate the fork stack as sequential cherry-picks onto a new upstream tag.
# Usage: simulate-picks.sh <old-upstream-tag> <fork-ref> <new-upstream-tag> [out-dir]
# Output: <out-dir>/sim.tsv with one row per fork commit: index, sha, clean|CONFLICT, files, subject.
# Conflicts are committed with markers so later picks keep going, so later rows
# can cascade from earlier ones. Treat the conflict count as an upper bound.
set -euo pipefail

old_tag=${1:?old upstream tag}
fork_ref=${2:?fork ref, use refs/heads/... to avoid tag/branch name clashes}
new_tag=${3:?new upstream tag}
out=${4:-/tmp/port-sim}

repo=$(git rev-parse --show-toplevel)
scratch="$out/scratch"
mkdir -p "$out"
rm -f "$out/sim.tsv"

base=$(git -C "$repo" merge-base "$fork_ref" "refs/tags/$old_tag")
git -C "$repo" worktree add --detach "$scratch" "refs/tags/$new_tag" >/dev/null
trap 'git -C "$repo" worktree remove --force "$scratch" >/dev/null 2>&1 || true' EXIT

cd "$scratch"
# Per-command identity: a bare `git config` here writes the repository's
# shared config (worktrees have no private config file), and every later
# commit in the repository and its worktrees inherits it.
sim_id=(-c user.name=sim -c user.email=sim@example.com)

n=0
for c in $(git -C "$repo" rev-list --reverse --no-merges "$base..$fork_ref"); do
  n=$((n + 1))
  subj=$(git -C "$repo" log -1 --format=%s "$c")
  if git "${sim_id[@]}" cherry-pick --allow-empty --keep-redundant-commits "$c" >/dev/null 2>&1; then
    printf '%s\t%s\tclean\t\t%s\n' "$n" "${c:0:10}" "$subj" >>"$out/sim.tsv"
  else
    files=$(git diff --name-only --diff-filter=U | tr '\n' ',')
    printf '%s\t%s\tCONFLICT\t%s\t%s\n' "$n" "${c:0:10}" "$files" "$subj" >>"$out/sim.tsv"
    git add -A >/dev/null 2>&1
    git "${sim_id[@]}" -c core.editor=true cherry-pick --continue >/dev/null 2>&1 ||
      git "${sim_id[@]}" commit --allow-empty -qm "sim-conflict $c" >/dev/null 2>&1
  fi
done

awk -F'\t' '{c[$3]++} END {for (k in c) print k, c[k]}' "$out/sim.tsv"
