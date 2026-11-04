# Cut and Map

Before any patch is applied: cut the port branch from the release tag, map the fork stack onto it, and give every fork commit one disposition. Read-only and branch-local. Nothing here pushes or publishes.

---

## 0. Preconditions

- The upstream release tag exists and is the one you mean. Confirm against the remote, not just the
  local tag:
  `git ls-remote upstream refs/tags/v0.32.0` and `git rev-parse v0.32.0`.
- The fork's current state is captured before you touch anything: `git ls-remote origin` for the
  deployed branches and their SHAs, and the SHA of the fork tip you are porting from. "Before" has
  to be provable later.
- The fork branch you are porting from is *not* the branch production is currently serving, and you
  are not going to move the fork's default branch during the port.
- The upstream image for the new tag may not exist yet (upstream published v0.31.0's image roughly
  34 hours after its tag). Check the registry before planning the image step. Do not substitute a
  nightly, the patch base would differ and every artifact in the port would be unreproducible.

---

## 1. Cut the branch

- **Cut straight from the new release tag. Never rebase the fork's history onto it.** Release
  branches diverge. `v0.31.0` is not an ancestor of `v0.32.0`, and they share only a merge-base
  (`3d5f4d4cd5` for the 0.31→0.32 cut). A port is a re-authoring exercise, not a rebase.
- **Cut in a separate worktree** so the primary checkout keeps serving the live branch:
  `git worktree add .claude/worktrees/v0.32.0z -b v0.32.0z v0.32.0`.
- **Fetch upstream with full history. Never `--depth=1`.** Every later step depends on ancestry:
  `merge-base --is-ancestor`, `git log <tag> --grep`, per-commit diffs against parents, and
  cherry-pick three-way merges. A shallow fetch silently breaks all of them.
- Record two SHAs in the plan immediately: the **cut point** (the release tag's SHA) and the
  **merge-base** with the fork's previous base. Both are load-bearing for the rest of the port.
- **Publish nothing.** No push of the new branch, no default-branch flip, no `0.32.0z` or `latest`
  tag. Build the local image under an rc tag (`0.32.0z-rc`) so the published tags cannot move even
  by accident.
- If the fork's history needs rewriting *before* the port (scrubbing, squashing, restructuring),
  do it in an isolated mirror clone with **range-limited** rewrites. An unlimited rewrite lets text
  replacements walk into upstream ancestors and diverge the whole ancestry. The 0.29 scrub matched
  a word in a 2019-era upstream comment and rewrote upstream commits.
  Verify merge-bases against the tags *before* pushing anything.
- Keep the plan file itself **tracked or backed up**. A plan living in an untracked
  `docs/superpowers/plans/` directory was destroyed once mid-port by a `git clean`. Reconstruct it
  from edit history and keep the backup outside the worktree. Ban `git clean` in every dispatch.

---

## 2. Establish topology before mapping

- `git merge-base --is-ancestor <old-base> <new-tag>`, if it returns non-zero, the release
  branches diverged and this is a port. If it returns zero, a rebase is possible and far cheaper.
  check that first, don't assume.
- `git merge-base <old-base> <new-tag>`, record the SHA. It is where the fork's own patches and
  upstream's drift separate, and it is the correct left-hand operand for drift measurement.
- Per-branch commit counts (`git rev-list --count <a>..<b>`) for both the upstream delta and the
  fork delta, so the plan can quote an exact stack size and later verify a commit count at each
  task gate.
- A trial merge of the whole stack (`git merge-tree`) is worth running once to count conflicted
  files, but treat the count as a floor, not a ceiling. In the 0.32 port, 6 of 51 commits
  conflicted while the map turned up several silent, conflict-free hazards (duplicate method
  definitions, inverted feature guards, an upstream capability gate with no GPU check).

---

## 3. Enumerate and classify the fork stack

- Take the fork's commits as a **bounded range**: `git log --oneline <old-base>..<fork-tip>`. Keep
  the list numbered and stable for the whole port. Every later reference in the plan should be a
  number from that list, not a paraphrase.
- Classify each commit before deciding anything:
  - **upstream cherry-pick**, a fork commit carrying an upstream PR number. Verify with
    `git log <new-tag> --grep='<PR#>'`.
  - **cohort-sync commit**, exists only to reach a particular upstream tree. Dies on a new base.
  - **fork-local feature**, the real payload.
  - **self-cancelling series**, implemented, then reverted. Prove zero residue with grep before
    dropping it.
- **Absorption is a content question, not a SHA question.** A cherry-picked commit has a fresh SHA
  in the fork, so `merge-base --is-ancestor` will never prove the fork's copy matches upstream's.
  Search the new upstream tree for the fork's distinctive symbol, warning string, or config key.
- For each fork-local patch, ask the operational question: *if upstream already fixed this, would
  the fork's copy be a no-op, a duplicate, or a conflict?* The three answers lead to three
  different dispositions (DROP / KEEP-REDUNDANT / ADAPT), and only the third needs work.
- **Do not trust "unchanged upstream ⇒ applies clean."** Verify: two files can be byte-identical
  across the cut while their *neighbours* in the same commit drifted, so the hunk context no longer
  matches. Check the pathspec of every file in the commit, not the file the plan names.

---

## 4. Measure drift

- Build a **drift map**: for every file the fork touches, measure upstream churn since the
  merge-base with `git diff --numstat <merge-base>..<new-tag> -- <path>`. Rank files by churn.
  Highest-churn files are where composition defects live, in the 0.30 port, three of twelve
  patches landed in one `scheduler.py` that drifted +351/-129, and the file where each patch was
  individually correct but jointly wrong.
- Recompute every number **programmatically from git** and snapshot the output. Hand-written
  tables go stale and transpose: one port's table transposed two tiering rows and invented a
  "new" file that was byte-identical, while omitting an entirely new upstream tier directory.
- Mark explicitly, per file: **deleted upstream**, **renamed/moved upstream**, and **new upstream**.
  Renames are the classic silent breakage, upstream deleting `scripts/` turns a file move into a
  location conflict. Upstream moving a plugin base class from `qwen4_exp/nvidia/` to `common/`
  breaks every source-hash pin that references it.
- **Verify pinned-file identity by blob SHA across all three refs** (old cohort target, fork tip,
  new base): `git rev-parse <ref>:<path>`. A SHA256 source pin survives only if the blob is
  unchanged.
- **Re-pinning a hash is not a fix.** When upstream moves a pinned file, re-pinning the new hash
  hides a real behavioural break if the file's semantics changed. In the 0.32 port, upstream made
  the PLE prefetch buffer lazy and dtype-aware. The plugin's constructor raises `AttributeError` on
  every int4 layer, and re-pinning the two moved base classes would have made the pin pass while
  the plugin stayed broken. Read the moved file before you re-pin it.
- Also measure the **toolchain pins** your patches assume (torch, flashinfer, plugin deps) in the
  *new* base and the *new* image, from the environment, not from memory. The fork's authored
  flashinfer target was one patch-level off from what the new venv and the new image actually
  shipped.

---

## 5. Classify the hazards git will not report as conflicts

A clean merge is not a correct merge. Check each of these explicitly for every file the fork owns:

- **Duplicate definitions from auto-merge.** Merging an upstream file that grew a method with a fork
  file that grew the same method yields two definitions. Python silently keeps one. In the 0.31
  port, `cpu/manager.py` ended up with two `on_request_finished` methods, upstream's won, and the
  lookup pins were never released, the CPU tier eventually refused every store. Run a
  duplicate-definition lint (ruff) on every merged file, not just the conflicted ones.
- **Feature-detection guards that invert.** A plugin guard written as "upstream has no knob, so
  install ours" that literally tests `hasattr(module, "symbol")` agrees with the old base and
  disagrees with the new one. Out-of-tree code fails this silently because the branch that changes
  behaviour is the one nobody exercised.
- **Upstream capability gates with no hardware check.** Upstream's new fused QSA writer is on by
  default, gated only on rotary/head-shape, and uses an FP8 type Triton cannot compile below
  SM89. On GPUs below SM89 this fails loudly at graph capture *and* silently blinds the
  calibration collector and clip counter (their unit tests still pass, because they call the
  helpers directly). The map must surface upstream features the fork never had to think about.
- **Unit refactors.** Upstream moving the CPU manager from blocks to chunks makes fork code that
  divides by `self._num_blocks` wrong at runtime, while the fork's own tests, which construct the
  base class directly, keep passing. Name the new seam explicitly in the plan.
- **New upstream pre-commit hooks that reject fork-owned files.** A new `check-test-tethering` hook
  rejects test files no Buildkite job runs. Each rejected file needs an allowlist line in the
  commit that adds it.
- **Silent behaviour changes under a flag.** Upstream routing all all-reduces through the custom
  path *when `VLLM_BATCH_INVARIANT` is set* interacts with the fork's own all-reduce probe. Check
  the upstream gate conditions before carrying a fork probe forward unchanged.

---

## 6. Decide the mechanic before writing the plan

Pick **one** mechanic per patch class. Mixing two in one plan was the single largest source of
defects found in a deep plan review.

- **Per-commit-range extraction (default).** `git diff <parent>..<commit> -- <pathspec>` isolates
  one patch with no subtraction and no hunk surgery. Two-tag diffs plus interactive `git add -p`
  splitting assume a hunk-level separability the fork's history does not have: in the 0.30 port,
  four hunks interleaved a feature with the fix that completed it, so the split was not achievable
  hunk-wise and the "checksum-only" path referenced state that only existed in the feature.
- **Cherry-pick** when the commit is an upstream cherry-pick, or when it applies cleanly and you
  want to keep upstream's authorship and message.
- **Fresh squashed re-author** when upstream rebuilt the file underneath the patch, when the commit
  is one fragment of a saga, or when the intermediate tip would be red. Name the fork commits it
  subsumes in the message.
- **Branch straight from the new tag and apply fresh commits** rather than replaying the fork's
  history, when the fork's history contains revisions that were later undone. This prevents a
  superseded implementation from returning with the replay.
- **Never use `git add -p` to split a commit whose hunks are interleaved.** Split it upstream, by
  authoring the two commits separately against the target base.

---

## 7. Disposition table

Every fork commit gets exactly one row, decided at map time:

| Disposition | Meaning |
|---|---|
| `DROP-ABSORBED` | Upstream ships the fix. Verify by content, not SHA. |
| `DROP-SELF-CANCELLED` | The fork reverted it. Prove zero residue by grep. |
| `DROP-COUPLED` | Existed only to reach an upstream cohort tree. |
| `DROP-UPSTREAM-OWNED` | The file belongs to upstream (templates, licences, shared docs). Leave it alone. |
| `KEEP-APPLY-CLEAN` | Upstream did not touch the pathspec. Confirm per-path, not per-commit. |
| `KEEP-REAUTHOR` | Upstream restructured the boundary. Rewrite against the new seam. |
| `ADAPT` | The interface changed. The behaviour is still wanted. |
| `SPLIT` | One commit carries two independent payloads. Check the split's own adjacency. |
| `FOLD` | A fix-of-fix that lands inside its target's commit. |
| `NEW` | Upstream's new code needs a guard the fork must add. |

- **Folding is adjacency-constrained.** Fold a fix-of-fix into the commit it completes only when
  the two are adjacent, or when no intervening commit touches the same files. Non-adjacent folding
  is banned because a non-adjacent fixup diff encodes a *later* tree and silently lands work the
  fork had not yet decided. Reordering a pick next to its target is safe only when nothing in
  between touches the same files, verify per fold, and leave any fold that fails as its own pick.
- **A split's hunks must be checked for overlap with the rest of the stack**, not just against the
  commit being split. In the 0.32 port, one commit's calibration hunk overlapped nothing, while
  its plugin hunks collided with two later commits, so the split moved those hunks into the
  plugin group instead.
- **Documentation tables are shared property.** If a fork patch touches a doc table upstream also
  edits, splice the genuinely new rows into the target branch's table. Do not take the whole file
  from either side.
- **Defects on the source branch are not port defects.** Enumerate them during the map, keep the
  port behaviourally equivalent to the source branch, and fix them afterwards in separate, clearly
  labelled commits. Fixing them mid-port breaks equivalence and destroys your ability to attribute
  a regression to the port rather than to the fix.
- **Anything that needs a real boot to disposition, boot before you commit to the disposition.**
  A 5-commit exclusion patch looked absorbed in the 0.30 map. The stock boot proved it was not
  (the assert fired on every worker) and the disposition flipped from `DROP` to unconditional
  `KEEP`. Probe with the fork's own filters disabled so you are testing upstream's behaviour.

---

## 8. Plan-level discipline (before execution)

- **Review the plan adversarially before picking anything.** A deep review of the 0.30 plan
  returned 41 confirmed findings and the largest single source of defects was the plan's own port
  mechanic. A second review of the *revision* found two high-severity defects the revision had
  introduced itself. Run at least two rounds. The second round exists to catch wave-one mistakes.
- **Make the disposition table's claims verifiable.** Every `DROP-ABSORBED` row needs a grep or
  symbol check whose failure is loud. Anchor the greps to fork-specific attribute names: a
  grep for `alignment|shave|shift` matched sixteen lines of live `alignment_chunk_count` feature
  code, and executed literally it deleted valid coverage for a feature that still existed.
- **Record a baseline before the first pick, using the exact invocation you will use at the end.**
  - `pytest` treats a missing path as a *usage error* and runs nothing, so a baseline command that
    names a file absent from stock silently produces an empty result.
  - `.venv/bin` must be on `PATH` for `python -m pytest` to find `vllm` as a subprocess binary.
    without it the server-booting tests error at spawn and contaminate the baseline.
  - Fork-only test files do not exist on stock, so the stock baseline cannot see them. Snapshot
    their collected test names and pass/fail from the old fork worktree and diff against that.
- **Merging old test files into the new base is a merge, not a checkout.** `git checkout
  <old-tag> -- <testfile>` overwrites upstream's version of a file upstream also changed, silently
  deleting upstream regression tests (13–17 functions lost from one scheduler test file, plus
  3–4 from a manager test and 3 from a shared-region test). Merge by hand and add a
  collected-test-count assertion so a deletion fails loudly instead of disappearing.
- **Scope each task's green gate to the tests that task owns.** A task that gates on a merged test
  file cannot pass at its own endpoint when the file also carries the next task's cases.
- **Write documentation rows as pending, then amend.** The branch row in the README goes in as
  "pending verification" in the commit that creates the branch, and the verifying commit amends it
  in place. Do not commit a README that claims a verification that has not run yet.
- **Make every commit self-describing.** Authored commits name the fork commits they subsume and
  the upstream PRs they correspond to. Cherry-picks keep upstream's message. Provenance in the
  message and a README patch table is the substitute for git lineage, which cherry-picks destroy.
- **Keep a ledger of state** (tasks complete, deferred minors, open questions) in the worktree
  itself. Session isolation means an external workspace will not be visible to the next
  implementer.

---

## 9. Cutting the shippable artifact

If the fork's image is built by overlaying a patch on the upstream image, generate that patch from
the **pushed remote tip**, not the local working tree:

- `git diff <upstream-tag>..origin/<branch> -- <pathspec>`. The right operand is `origin/<branch>`
  so the artifact reflects exactly what the image represents, and local churn cannot leak into it.
- **Scope the pathspec to paths that exist in the target image**, `-- vllm/ <plugin-dir>/`. A
  full-repo diff carries files the image does not have (Dockerfile, `scripts/`), and `patch -p1`
  then fails or scatters stray paths.
- **Overlay a patch. Never COPY the fork's `vllm/` tree over the image's.** Copying replaces the
  CUDA-aware build of the C extensions with a source tree.
- **Install out-of-tree plugins with `pip install <dir>`, not `COPY <dir>/`.** A COPY of the plugin
  *repository* nests the package (`ple_int4/ple_int4/__init__.py`), which Python reads as an
  implicit namespace package with no `__init__` and no entry point, the plugin silently never
  loads. `pip install` flattens the layout and registers the entry point the plugin was designed
  around.
- Keep the regenerated patch file out of the committed tree, but do **not** put its directory in
  `.dockerignore`, that makes the `COPY` step fail with "forbidden path".
- Expect layer-cache behaviour: the patch layer hits only when the `git diff` output is
  byte-identical, and a registry digest commits to the exact layer set, so floating dependency
  versions change the digest even when the application code is unchanged. Pin deployments to
  digests when you want bit-reproducibility.

---

## 10. Verification gates for the map

Before execution starts, all of these must be true:

- Topology recorded: cut-point SHA, merge-base SHA, upstream commit counts, fork commit count.
- Every fork commit has exactly one disposition, and every `DROP` has a check that would fail if
  the drop were wrong.
- The drift map's numbers were produced by a git command whose output you can re-run, not
  transcribed.
- Every pinned-file SHA was re-verified across all three refs, and every re-pin has a note about
  whether the moved file's *semantics* changed.
- The plan states, per task, the exact commit-count gate it must satisfy at its end.
- At least one boot-level probe has been run against stock upstream where a disposition depended
  on runtime behaviour, with the fork's filters disabled.
- The baseline exists, uses the same invocation as the final validation, and includes the
  fork-only test files.
- Nothing has been pushed, no image tag has moved, and the fork's default branch is unchanged.

---

## Worked examples (one line each)

- **0.29 → 0.30:** upstream rebuilt the offloading boundary (`KVCacheGroupRole`, `OffloadingConfig`,
  `HISPARSE_INDEXER`), so a 5-commit exclusion patch looked absorbed but was not, the stock boot
  asserted on every worker and the disposition flipped.
- **0.30 → 0.31:** upstream shipped the misaligned-group exclusion, so the stack dropped from 41
  commits to 32, but only after a CPU-only probe plus a static check confirmed it.
- **0.31 → 0.32:** upstream absorbed none of 51 commits. The stack went to 42 via seven verified
  folds, one packaging merge, one new capability gate, and six real adaptations.
- **Every port:** the merge that looked clean was where the interesting defect lived.
