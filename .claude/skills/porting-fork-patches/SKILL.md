---
name: porting-fork-patches
description: Port the fork's patch stack (the vX.Y.Zz branches) onto a new upstream vLLM release. Use when a new upstream release is out and the fork needs a new vX.Y.Zz branch, when mapping drift or deciding which fork commits upstream absorbed, when writing or reviewing a port plan, when resolving conflicts while applying the stack, when verifying or releasing a port, or when asked how a previous port was done.
---

# Porting the fork patch stack

Each upstream release gets a new fork branch `vX.Y.Zz`, cut from the release tag. Release tags sit on divergent branches, so the job is a port anchored on the merge-base, never a rebase. Every stage ends in gates. Do not start a stage until the previous stage's gates hold.

Read the reference for the stage you are in. Each ends with a gate list and one-line worked examples.

| Stage | Reference | Output |
|---|---|---|
| 1. Cut and map | `references/cut-and-map.md` | Worktree at the release tag, drift map, one disposition per fork commit |
| 2. Plan and review | `references/plan-and-review.md` | Reviewed plan with a decision ledger, approved before execution |
| 3. Execute | `references/execution.md` | Branch of green commits, GPU-owed ledger, rc image |
| 4. Verify | `references/verification.md` | Baseline diff, red-proofs, artifact and runtime proofs |
| 5. Release | `references/release.md` | Pushed branch, tags and image, closed release ledger |

## Stage 1 quick path

```bash
git fetch upstream --tags            # full history, never --depth
git merge-base --is-ancestor <old-tag> <new-tag> || echo "divergent: port, not rebase"
git worktree add .claude/worktrees/<new-branch> -b <new-branch> refs/tags/<new-tag>
.claude/skills/porting-fork-patches/scripts/simulate-picks.sh <old-tag> refs/heads/<old-fork-branch> <new-tag>
```

The fork branch and its tag share a name, so pass full ref paths. The simulation script writes `sim.tsv` with one row per fork commit. Treat the conflict count as an upper bound because conflicts cascade, and as a floor for hazards because clean merges hide the worst defects.

Then measure drift per file the fork touches (`git diff --numstat <merge-base>..<new-tag> -- <path>`), list files changed on both sides, and check upstream log titles for the fork's areas to find absorbed patches. Absorption is a content question, not a SHA question.

## Rules that held in every port

1. **A clean merge is not a correct merge.** The worst defect of every port sat in a file that merged without conflict: duplicate method definitions that Python resolves silently, feature guards that invert, upstream capability gates with no hardware check. Run a redefinition lint on every merged file and read upstream's new code paths the fork never had to think about.
2. **Disposition every fork commit exactly once** (`DROP-*`, `KEEP-APPLY-CLEAN`, `KEEP-REAUTHOR`, `ADAPT`, `SPLIT`, `FOLD`, `NEW`) and give every `DROP` a check that fails loudly if the drop is wrong. Boot stock upstream with the fork's filters disabled before dropping anything whose failure is runtime-visible.
3. **Keep ported commits behaviorally equivalent to the source branch.** Adaptations are separate, labeled commits. Defects inherited from the source branch are fixed afterwards in their own commits, each with a red-to-green proof.
4. **Fold only adjacent commits**, or commits where no intervening commit touches the same files. Never split interleaved hunks with `git add -p`.
5. **Re-pinning a hash is not a fix.** Read the moved file and confirm its semantics before updating a pin.
6. **Record the baseline before the first pick**, with the exact invocation the final run will use, suites run serially, failures decomposed by class: red on stock, GPU-owed, regression. Gate on collected test names and counts as well as on the failure diff.
7. **Merge old test files into the new base, never check them out.** A checkout silently drops upstream's regression tests for that file.
8. **Verify the plan against the repository.** Every claim in a plan or fix wave carries a re-runnable check. Review at least two rounds, the second scoped to regions the fix wave changed, and stop when a round's confirmed defects are all regressions from the previous wave.
9. **A fresh reviewer per task who re-runs the gate.** The implementer's report is not evidence.
10. **Nothing outward-facing without the owner.** No push, no moving tag, no default-branch flip, no `latest`. Build under an rc tag. Generate the image diff from the pushed tip, scoped to image paths, and overlay it. Never copy the fork tree into the image. Install plugins with `pip install <dir>`.
11. **Keep the plan and ledger safe.** Back the plan up outside the worktree and ban `git clean` in every dispatch. The ledger lives in the worktree and records every ruling, deferral and GPU-owed item with a name, a reason and an owner.
12. **Treat commit messages and README prose as public.** No hostnames, addresses, names, topology or process narration. A message-only rewrite must leave every tree hash unchanged.

## Unsettled across ports

The ports disagreed on these. Until the owner rules, use the working default and record the choice in the plan's decision ledger.

| Question | Working default |
|---|---|
| Is an upstream-side or static claim enough for `DROP-ABSORBED`? | No for anything runtime-visible, boot stock upstream. |
| Port mechanic | Simulate cherry-picks to map drift, then pick a mechanic per patch class and do not mix within a class. |
| Fix source-branch defects during the port? | No, land them after as labeled commits. |
| Baseline with known deterministic errors | Fix the invocation and re-record before the first pick. |
| Is a split admissible when its hunks are not adjacent to their new home? | No, leave it whole. |
| Plugin packaging | `pip install <dir>` in every image. |

## Before starting a port

Use the previous port's plan (`docs/superpowers/plans/*-port.md`) as the structural template. Check the registry for the upstream image at the new tag, which has appeared more than a day after the tag. Never substitute a nightly.
