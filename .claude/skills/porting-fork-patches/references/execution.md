# Execute the Port

Mechanics of applying the stack: workspace, execution mode, per-class mechanics, conflict resolution, landing commits green, image build, publish, history scrub and the post-port defect series.

---

## 0. Preconditions

- The plan exists, has survived at least two adversarial review rounds, and every decision in its
  ledger is **approved**, not merely reasoned about. An open decision is a licence to guess, and a
  guess in the middle of a forty-commit port is how the branch diverges from its own contract.
- The disposition table is final. Every fork commit has exactly one disposition, and every
  `DROP-*` has a check that would fail loudly if the drop were wrong.
- The branch is cut, the worktree exists, and the tree is clean. The upstream release tag is the
  one you mean, and the fork's default branch is untouched.
- Nothing is published. No branch push, no `0.32.0z` tag, no `latest` tag, no default-branch
  flip. The build script must not be able to stamp a moving tag as a side effect of an ordinary
  build.
- The ledger is in the worktree, next to the plan. It is the only channel that survives between
  tasks, and every ruling you make during execution goes into it.
- You have a known-good image of the branch you are replacing. Almost every question that arises
  mid-port reduces to "is this different from what is already in production", so it has to exist
  before the first question arrives.

---

## 1. Prepare the execution workspace

- **One worktree per port, never the primary checkout.** The primary checkout is serving the live
  branch. The port worktree is disposable:

  ```bash
  git worktree add .claude/worktrees/v0.32.0z -b v0.32.0z v0.32.0
  ```

- **Record the topology before anything else touches the tree.** These four numbers are
  load-bearing for the rest of the port and cost nothing to capture:

  ```bash
  git rev-parse v0.32.0                          # cut point
  git merge-base v0.31.0 v0.32.0                 # merge base
  git rev-list --count v0.31.0..v0.32.0          # upstream delta
  git rev-list --count v0.31.0z..origin/v0.31.0z # fork delta
  ```

- **Build the environment in separate steps, and prove the tree survived.** Install the tree, then
  install the extras (the audio extra in particular pulls transitive dependencies), and diff the
  package location before and after. An extras install that relocates the tree is a silent
  substitution, and it has happened.

- **Pin the transitive dependencies an extras install would otherwise drift.** Installing
  `vllm[audio]` silently downgraded NCCL from 2.30.7 to 2.29.7 in one port. Pin the version
  explicitly and assert it in the build.

- **Capture the baseline with the exact invocation the final validation will use**, including the
  fork-only test files that stock cannot see. Run the suites serially, two GPU suites running
  in parallel put their contention failures into the baseline and it looks like a clean reference
  that it is not. Decompose the baseline by failure class, not by count.

- **Ban `git clean` in every dispatch, and ban touching `docs/` and `.superpowers/`.** A plan file
  living in an untracked `docs/superpowers/plans/` directory was destroyed mid-port by a `git
  clean` and had to be reconstructed from edit history. Keep a copy of the plan outside the
  worktree as well.

- **Use absolute paths in every dispatch.** Agent threads reset their working directory between
  shell calls, and a relative path that worked in one call silently resolves somewhere else in
  the next.

---

## 2. Choose the execution mode, and honour it

- **Subagent-driven development is the default for a port of this size.** One fresh agent per
  task, a review between tasks, independent patches each gated on their own tests. The
  freshness is the point: an agent that implemented task 4 is the worst possible reviewer of task
  4, and the 0.31 port's one critical defect (duplicate `on_request_finished` in
  `cpu/manager.py`) was invisible to the implementer and to a whole-branch diff review alike.

- **The dispatcher must prove the dispatch happened.** In one run the final review slice was
  never dispatched because the queue stalled. The port sat at "review pending" with nothing in
  flight. After every dispatch, confirm the worker is running. Do not infer progress from
  silence.

- **Give each worker the plan text, the base SHA, the tip SHA, and the acceptance command.
  Nothing else.** A worker that inherits conversational context will also inherit the
  assumptions that turned out to be wrong.

- **The reviewer is not the implementer and does not inherit the implementer's report.** It
  re-runs the gate itself. "The implementer says it passes" is not evidence.

- **Model the intermediate state at every task boundary.** A task must leave the tree green at its
  own tip. A red task tip hands every later task a broken baseline, and baseline diffing stops
  producing signal.

- **Watch for the plan file disappearing.** If the plan is untracked in the worktree, back it up
  outside the worktree before the first task and re-check that it exists after each task that
  touches paths near it.

---

## 3. Apply the stack: pick a mechanic per patch class

The disposition table from Section 1 assigns each commit a disposition. Each disposition has a
matching mechanic. Do not use one mechanic for the whole stack.

- **Contiguous fork range, extracted as a diff.** The default for a coherent series of fork-local
  commits:

  ```bash
  git format-patch --stdout <fork-base>..<fork-tip> > /tmp/stack.patch
  ```

- **Cherry-pick**, only for commits that are themselves upstream cherry-picks or apply cleanly
  with no surrounding context drift. Cherry-pick destroys merge-base evidence, so a fix that
  lands as a new SHA looks absent from history. Verify fixes by content, not by lineage.

- **Fresh squashed re-author**, when upstream rebuilt the file underneath the patch or the series
  is a saga whose intermediate states were never meant to be reviewable. Re-authoring is what the
  0.30 port did for all twelve of its fork-local patches: extract the diffs, squash them, and land
  them as new authored commits. No cherry-pick, no rebase, no resurrected reverted history.

- **Straight branch** for a revision that upstream undid and the fork wants to keep.

- **Never `git add -p` to split interleaved hunks.** A hand-picked hunk set is unreviewable, and
  the two hunks you wanted are the two you will get wrong. Split by file or not at all.

- **Fragmented chains and adaptations land as a single net-change commit over the adjacent
  diff.** Clean, non-conflicting picks land untouched. An authored commit names the fork commits
  it subsumes, so the history stays readable and the equivalence check has something to compare
  against.

### 3.1 `git apply` is not atomic

- **A multi-file patch can apply partially and still exit zero.** One port applied a four-file
  patch, got a clean exit code, and landed three of the four files. Treat a patch application as
  all-or-nothing: `git apply --check` first, then apply, then **always read `git status --short`
  back** and compare against the file list you expected.

- When a patch cannot apply whole, apply the code hunk alone and amend the surrounding commit
  manually. Do not let a partial apply sit in the tree where the next agent will read it as
  intentional.

### 3.2 Folding, squashing, and commit-shape hygiene

- **Fold only across adjacent commits.** Reordering a fixup past an intervening commit that
  touches the same files encodes a later tree into an earlier diff. The 0.32 port folded seven
  commits and verified the adjacency condition for each with a helper script. Two of the folds
  only turned out safe because the hunks were disjoint, and the plan never said so.

- **Squash a documentation series into a single commit.** Nine README-cleanup commits were
  collapsed into one so the history does not advertise the editing process.

- **Use `git commit --fixup` plus `--autosquash`**, not `git fixup -c`. The interactive variant
  had an author-date bug that rewrote authorship on the squashed commit.

- **Keep the trailer block last.** A paragraph appended below the trailers stops trailer parsing,
  which breaks attribution and compliance. Repair by moving the paragraph above the block. Where
  the original commits never had trailers, surface the count and leave them alone rather than
  adding sign-offs that were never there.

- **A message-only rewrite must move no content.** After repairing trailers, verify that every
  tree hash is unchanged. The hashes are the proof.

---

## 4. Resolve conflicts without inventing behaviour

- **Simulate the merge before you apply it.** `git merge-tree` (or a throwaway merge in a scratch
  branch) shows you which files upstream rebuilt underneath the patch and which hunks are
  actually contested. In one round this refuted the highest-severity plan-review finding.

- **The merge that looks clean is where the interesting defect lived.** Every one of the three
  ports found its worst problem in a file that merged without conflict: a duplicated
  `on_request_finished` in `cpu/manager.py` (0.31), an upstream-deleted `scripts/` directory
  that orphaned the fork's build script (0.31), a fused QSA writer that silently nulls key and
  value for the calibration collectors (0.32).

- **Auto-merge produces duplicate definitions that Python resolves silently.** When upstream and
  the fork both define the same method on the same class, the merge is "clean" and the later
  definition wins. In `cpu/manager.py` the upstream definition won, so the fork's lookup pins
  were never released and the CPU tier eventually refused all stores. Hand-merge these files, and
  add a lint check for the class of artifact: `ruff` catches a redefinition, and it should be in
  the tree permanently.

- **When a file changed on both sides, take a signature union rather than picking one.** The QSA
  kernel conflicts in the 0.30 port were resolved by keeping upstream's `output_gate` and the
  fork's `k_scale`/`v_scale` in the same signature. Picking a side silently drops a parameter.

- **When upstream moved a file, follow the move and re-verify the semantics.** Upstream relocated
  the PLE base classes into `common/`, which broke a plugin's `sha256` pin. Re-pin inside the
  specific commit that owns the pin, and check that the moved file's *semantics* did not change
  before re-pinning the hash. A pin that is re-hashed without a semantic check is a licence to
  ship whatever the file now contains.

- **When upstream deleted a directory, relocate rather than resurrect.** Upstream removed
  `scripts/`. The fork's build script moved to `tools/`. Bringing the directory back puts you
  one upstream merge away from the same fight again.

- **Watch for the inverted feature guard.** The plugin's offload hybrid patch was installed under
  a `hasattr` guard on `get_offloading_group_ids`. On the old base the symbol was absent, the
  guard passed, and the custom logic installed. On the new base the symbol exists, the guard
  fails, and the patch is silently not installed, the exact opposite of the intended outcome.
  Any guard of the form "if the new API is absent, do the fork's thing" inverts across a port
  that adds the API. Find these before the port, not after the first silent wrong answer.

- **A file that upstream "did not change" may still have drifted neighbours.** Do not trust a
  clean apply as proof of no drift. Check the pathspec against the drift map from Section 1.

---

## 5. Land every commit green

- **Test-first. No exceptions, including for one-line fixes.** Write the test, watch it fail for
  the right reason, then fix. A test written and passed in one sitting proves nothing about the
  bug it claims to cover.

- **Never ship a red suite.** A task tip that is red poisons every later baseline diff. If a
  test cannot go green in this port, name it as expected-red with the reason attached, and name
  the real acceptance surface alongside it.

- **Classify every failure into exactly one of three buckets** before you diagnose it:
  already-red-on-stock, GPU-owed, or regression. The 0.31 port ended with 200 failures, which
  decomposed into 123 baseline-red-on-stock and 77 GPU-memory. None were regressions, but that
  was only knowable because the baseline was captured first.

- **Gate on collected test names and counts as well as on the failure diff.** A deleted, renamed,
  or skipped test produces no failure and is a silent regression. The strongest acceptance gate in
  any of the ports was a count of collected test names before and after a test file merge, which
  turned a silent deletion of thirteen upstream regression tests into a red line.

- **Merge fork-only test files into the new base. Never check them out from the fork.** Checking
  out a file replaces the upstream version and silently drops upstream's own regression tests for
  that file.

- **Prove the red signal twice.** Re-derive it by reverting the fix and confirming the test
  fails on the expected assertion, not on a collection error or an unbound mock. Where a
  tree-editing red proof is impossible, a `sitecustomize` shim that swaps a symbol at import time
  reproduces the failure without touching the working tree. Run a negative control: the same
  construction without the corruption must pass.

- **Assert the upstream behaviour you depend on.** If the fork relies on upstream rejecting a
  configuration, assert that rejection, so an upstream change that removes it fails your suite
  instead of quietly changing behaviour underneath you.

- **Every new test file needs to survive the upstream pre-commit hooks.** A new upstream hook,
  `check-test-tethering`, rejects test files that are not referenced by Buildkite. Add the
  allowlist entry in the same commit that adds the file, or the commit cannot land.

- **Expect the hooks to modify your files and review what they did.** SPDX headers were
  auto-added to two gate scripts that were not part of the approved scope. Revert them rather than
  shipping an unreviewed diff, and treat the revert as a deliberate commit. Run ruff and mypy
  yourself rather than discovering them at commit time.

- **Keep the commit-count gate.** Each task declares the number of commits it will land. Check
  it. A task that lands one commit more than planned usually means a fixup escaped the fold.

---

## 6. Keep the GPU-owed ledger

- **A green suite on a busy card proves less than it looks.** Every failure on a contended device
  is an out-of-memory failure until proven otherwise. The 0.31 port ran on busy devices and had
  roughly 160 GPU tests owed, QSA numerics, warmup compiles, the `kv_offload` transfers, none
  of which had ever opened a device.

- **Ledger them, do not block on them.** Each owed item names the suite, the count, the reason it
  is owed, and where it will be discharged. A port that refuses to finish without a GPU window
  never finishes.

- **Open a window deliberately.** Free a device on purpose rather than waiting for
  one to go idle. Run the suites serially while the window is open, and record what the window
  discharged.

- **Re-run on a free card before believing a failure.** The same nine `test_worker.py` CUDA
  failures appeared as regressions when another workload held the GPUs and as passes inside the
  window. A failure that disappears when the device is free was never a regression.

- **Device-zero CUDA starvation has a signature**: sixteen failures that all name device 0 and
  all read as allocation errors. Verify one of them on another device. If it passes, the sixteen
  are capacity, not logic.

---

## 7. Build the image

- **Retarget the base versions to the new upstream release, and check the registry before you
  plan around them.** The upstream image for a release tag appeared roughly 34 hours after the
  tag in one case. Do not substitute a nightly, the patch base would differ and every artifact
  in the port would be unreproducible. Write the wait into the plan rather than leaving it to
  the implementer.

- **Build under an rc tag locally.** `0.32.0z-rc` first, always. A published tag must not be
  able to move because someone ran a build.

- **Generate the artifact diff from the pushed remote tip, never from the local working tree**,
  and confirm the local tree has nothing uncommitted that would be missing from the artifact.

- **Overlay the patch onto the upstream base. Never copy the fork tree into the image.** A copy
  step pulls in `.git`, untracked files, and local build residue, and it silently ships whatever
  was in the working directory.

- **Install plugins with `pip install <dir>`, not `COPY`.** Copying a plugin directory into the
  image skips the entry-point registration, and the plugin then imports but does not register.

- **Scope the patch pathspec to the image paths.** A patch that touches build tooling outside
  the image produces a failed `COPY` ("forbidden path") or an image that does not match the tree
  you reviewed.

- **Expect the layer cache to lie.** The patch layer hits only when the `git diff` output is
  byte-identical, and a registry digest commits to the exact layer set, so floating dependency
  versions change the digest even when the application code is unchanged.

- **Inspect the registry manifest after pushing.** Push output is not proof that the tag you
  wanted points where you think it points.

---

## 8. Publish deliberately

- **Nothing moves until every gate has passed.** Verification runs on local state and an rc tag.

- **Push the branch first, then move the tags, then publish the image**, in that order, and check
  the remote listing before and after each outward-facing change.

- **A build that stamps the moving tag as a side effect promotes every stack that references it
  on the next recreate.** Verify the tag wiring against a stubbed container client, covering both
  the default-tag path and the explicit-tag path. The two have behaved differently before, with
  an explicit tag incorrectly stamping the moving tag.

- **Do not move `latest` and do not flip the default branch unless you were asked to.** When you
  are asked, do both deliberately and re-verify the remote state afterwards, including that the
  other published tags are untouched by name.

- **A local `latest` tag can be overwritten by an ordinary build.** Restore it from the registry
  before pushing, so a local rebuild never silently demotes the published tag.

---

## 9. Scrub the history before it becomes public

- **Treat commit messages and README prose as public artifacts.** They leak hostnames, addresses,
  client names, hardware details, fleet topology, internal gate vocabulary, prior-release
  anchoring, and the process narration ("Task 4", "the fix for D2", "has not booted").

- **Rewrite with `commit-tree` plumbing over a bounded range**, in an isolated mirror clone.
  Verify the merge-bases against the tags afterwards. An unlimited rewrite lets a text replacement
  walk into upstream ancestors and diverge the whole ancestry. The 0.29 scrub matched
  a word in a 2019-era upstream comment and rewrote upstream commits.

- **Preserve authors, dates, and commit count.** A scrub that changes the author list is a
  different operation from a scrub.

- **Verify tree hashes are unchanged** for every rewritten commit. Message-only, no content.

- **Squash the documentation cleanup series into one commit** and describe what the commit does,
  not how it was reached. A commit message that narrates the editing history is itself a leak.

- **Sweep the README a second time.** The first pass missed a residual count in a table row
  ("72 of 938 streams"). Run the claim check adversarially and accept that it finds wording
  that overstates the evidence.

- **Know what you cannot undo.** Force-pushed commits stay reachable on the hosting service until
  it garbage-collects. Say so plainly rather than implying the content is gone.

---

## 10. Land the defect series separately

- **Fixes to defects carried in from the source branch are their own commits, after the port.**
  The 0.32 port landed 42 port commits and then 11 defect commits, each with its own RED-to-GREEN
  proof. Mixing them into the port makes the equivalence check meaningless.

- **Choose the fix shape with the owner, in writing.** The degrade-to-miss crash was fixed by
  failing the corrupted request cleanly rather than retrying it, because the retry path needed
  new rollback machinery in the scheduler and the engine crash was worse than a single failed
  request.

- **Use the upstream channel when upstream already has one.** The "fail the request" shape was
  implemented on top of upstream's existing `kv_load_failure_policy` rather than by editing the
  core scheduler, and the alternative shape stayed available as a configuration switch.

- **Fix the locking lifecycle as a whole.** The registration lock leaked `.reglock` files into
  `/dev/shm`. The first fix, unlinking under the lock, had a race where one rank held a lock
  on a deleted file while another opened a new one. The working fix re-checks the file's inode
  after locking, on both the registering and the removing side, so two ranks can never hold the
  registration lock at once.

- **Do not widen an assertion to make it pass.** An "unpinned hit" assert that fired on a rare
  path was left narrow, because widening it would hide the bug it exists to catch. Narrow the
  assertion to the states that are truly impossible, or leave it and record the residual.

- **Deferred is a legitimate outcome, if it is written down.** The registration-lock deadlock
  stayed latent across two ports, ruled by the owner, with the fix design (a dedicated `.reglock`
  file, or POSIX byte-range locks) documented and not implemented. A deferral with a name, a
  reason, and an owner is a decision. A deferral that only exists in a conversation is a defect
  waiting to be rediscovered.

---

## 11. Execution gates

Before the branch is published and the image is promoted, all of these must be true and recorded:

- The cut point, merge-base, and both delta counts are recorded and re-derivable.
- Every plan task is complete, its gate passed, and its commit-count gate held.
- Every commit landed green, and every failure is classified as baseline-red, GPU-owed, or
  regression, with the GPU-owed list discharged or explicitly carried with a reason.
- The branch matches the deviation contract, and every divergence from the source branch is
  attributed to a ledgered adaptation.
- Fork-only files are byte-identical to the source branch where the plan says they should be.
- The working tree is clean, and the artifact diff comes from the pushed tip.
- The image was built from the reviewed tree, its registry manifest shows the intended tags, and
  the package location inside the image still points at the fork tree.
- The README and every commit message make no claim a gate did not support, and leak no
  hostnames, addresses, personal details, fleet topology, or internal process language.
- The published tags, the default branch, and the other tags were verified by remote listing
  after the last outward-facing change.

---

## Worked examples, one line each

- The multi-file patch that applied cleanly and still landed three files out of four, because
  `git apply` is not atomic and exits zero on a partial application.
- The clean auto-merge that produced two `on_request_finished` methods, where Python kept
  upstream's and the fork's lookup pins were never released, so the CPU tier eventually refused
  every store.
- The `hasattr` guard that inverted: absent symbol on the old base meant the fork's patch
  installed, present symbol on the new base meant it did not, and the failure mode was silence.
- The upstream commit that fused the QSA KV write into the pre-indexer launch, defaulting to an
  `fp8e4nv` path that will not compile below SM89 and that sets key and value to `None`, breaking
  the calibration and clip collectors without an error.
- The upstream typed `thinking` config that turned the fork's dict-style mapping into dead code
  that still compiled and still had tests.
- The plugin pin that broke because upstream moved the PLE base classes, fixed by re-pinning the
  hash inside the commit that owns the pin after checking the moved file's semantics.
- The 200 failures that decomposed into 123 already-red-on-stock and 77 GPU-memory, with zero
  regressions, knowable only because the baseline was captured serially before the first pick.
- The sixteen device-0 failures that turned out to be CUDA starvation and vanished on a free
  card.
- The `KV_DTYPE in (1, 3)` constexpr membership test that the Triton frontend refused to lower,
  fixed by splitting it into `== 1 or == 3` and proven on device.
- The history rewrite that walked into a 2019-era upstream comment and diverged the whole
  ancestry, and the range-limited rewrite that did not.
