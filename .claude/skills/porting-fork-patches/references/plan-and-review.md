# Plan and Review

Turn the disposition table into a plan an executor can follow without asking questions, attack the plan before it touches the tree, then review each task and the finished branch.

---

## 0. Preconditions

- The disposition table exists and every drop has a check that would fail loudly if the drop were
  wrong.
- The baseline exists and uses the exact invocation the final validation will use, including the
  fork-only test files that stock cannot see.
- Every fork commit has been replayed and has exactly one disposition.
- The ledger from the previous section is in the worktree and has been read by whoever writes the
  plan.
- The reviewer has the repository, not only the plan text. Every claim in this section is checked
  against the repository.

---

## 1. Anatomy of the plan

One file, dated, under `docs/superpowers/plans/`, named `YYYY-MM-DD-<topic>-port.md`. Commit it or
keep a copy outside the worktree. A plan that lived only in an untracked directory was destroyed
once mid-port by a `git clean` and had to be reconstructed from edit history.

Six top-level sections:

- **Goal and acceptance criterion.** One sentence each. For a port the acceptance criterion is
  behavioural equivalence with the source branch on the new base, except the listed deviations,
  plus zero regressions against the stock baseline.
- **Global constraints.** The port mechanic, the commit-shape rules, what must never be touched,
  what must never be published.
- **Decision ledger with stable identifiers.** Every judgement call gets a number such as D1 and
  D2. Decisions are approved before execution. An amendment references the original number so the
  history stays readable.
- **Task list.** Numbered, ordered, each landing a known commit count.
- **Deviation contract.** The places where the port deliberately differs from the source branch,
  each with a reason. This is the document the final branch review is checked against.
- **Final gates.** The checks that must pass before the branch is called done.

### The per-task template

Every task declares all nine of these. A task missing one is a defect the plan review will find,
and finding it late costs a rebase.

1. Base SHA it starts from, and the expected tip SHA it ends at.
2. The files it touches, by path, so two tasks cannot silently own the same file.
3. The numbered steps, each with the exact command to run.
4. The commits it lands and their count.
5. Its acceptance gate: the exact test invocation and the expected result.
6. The expected-red contract for tests in its files that must stay red.
7. What it owes later tasks: named symbols, markers, and files it creates.
8. What it consumes from earlier tasks.
9. The residual it knowingly leaves, and where that residual is tracked.

**Why:** The plan is dispatched to a worker that has no session context. Anything said out loud
during planning is lost at the next task boundary. Anything written into the task template
survives.

---

## 2. Writing the plan

- **Anchor every claim to a command that can be re-run.** Not "that file changed a lot" but
  `git diff --numstat <fork>..<upstream> -- <path>` produces the number. A number you can re-run
  is a fact. A number you transcribed is a rumour.
- **Anchor to content, not lineage, wherever the port will change SHAs.** Cherry-picking destroys
  `merge-base` evidence, so a fix that landed as a new SHA looks absent from history. Verify by
  content instead: grep for the distinctive symbol that only the fix introduces.
- **Describe the seam, not the edit.** For every adaptation the plan calls for, name the upstream
  function whose shape changed and the fork call site that breaks against it. That description is
  what a later task can be checked against, and it survives the SHA churn of the port.
- **Model the intermediate state at every task boundary.** A task must leave the tree green at its
  own tip. A task whose endpoint is red hands every later task a red baseline, and baseline
  diffing stops producing signal.
- **Make cross-task contracts explicit and bidirectional.** If task N waits on a marker task M
  leaves, task M must say it leaves one. A dangling reference to a marker that no task creates was
  found only in review, and it would have stalled the port in the middle of execution.
- **Compute every arithmetic gate in advance.** Expected commit count at each task, collected test
  count before and after a test merge, number of fold checks that must report safe. These become
  assertions later, and they are cheaper to derive now than to re-derive after a rebase.
- **Put the build and publish steps last and state their preconditions.** The upstream image for
  the new tag may not exist yet, sometimes for more than a day after the tag lands. Check the
  registry at plan time and write the wait into the plan rather than leaving it to the implementer.
- **Write the documentation row as pending, then amend it** in the commit that actually verifies
  it. Never commit a README that claims a verification which has not run.
- **Do not invent symbols.** A plan step that names a function which does not exist at that point
  in the tree invites the implementer to write it. If a plan says a step is a no-op because the
  code already matches upstream, say that instead.

**Why:** The plan review checks claims against the repository. A claim written from memory gives
the reviewer nothing to check, so the claim gets skipped, and the thing it was protecting ships
unverified.

---

## 3. Reviewing the plan before the first pick

### 3.1 Seed the review

- Do not let the reviewer rediscover the ground truth. Hand it the facts already established: the
  cut point, the merge-base, the disposition table, the baseline, the known suspect files.
- Name the suspect surfaces explicitly. The reviewer's budget then goes to reasoning instead of
  re-deriving facts you already paid for.
- The suspect surfaces in a port are always the same family: files upstream rewrote in the same
  release, files with two owners, seams where a fork API moved into a different module, and
  anything the disposition table marks as absorbed.

### 3.2 Run independent lenses

Run these as independent passes. One reviewer wearing every hat converges on the first framing it
accepts, and independent lenses disagree with each other. The disagreements are where the real
defects live.

- **Correctness.** Does the plan do what it claims, in the order it claims.
- **Consistency.** Does the plan contradict itself, and does its own helper script agree with its
  prose.
- **Contradictions.** Are two statements in the plan mutually exclusive.
- **Sequencing.** Does the task order work, and does the order of dependent changes work in both
  directions.
- **Simplification.** Is there a smaller correct option for any decision.
- **Duplication.** Does the plan describe the same work in two places that can drift apart.

### 3.3 Verify every finding adversarially

Every finding gets its own verification pass against the repository, not against the plan's
narrative.

- Require the verifier to reproduce the failure it claims. A finding that says two patches will
  conflict has to run the merge. In one round the top severity claim was refuted by a direct
  three-way merge simulation that showed the file survived unchanged.
- Count the refutations out loud. A round with 41 findings and 4 refutations is a working round.
  A round with zero refutations usually means the verifiers are rubber-stamping.
- The verifier sets the severity, not the finder, and findings get downgraded. A claimed memory
  leak was really a lost filename and a transient double count, because the kernel reclaims the
  memory once the last descriptor closes.
- A finding can be refuted by mechanism. State the mechanism in one sentence so the refutation can
  be re-checked.
- Disagreements between a task-level review and a branch-level review are not settled by seniority.
  Put both readings in front of the judge and settle them by mechanism.

**Why:** Finding quality is a function of verification quality. Unverified findings are guesses
with a severity label attached, and acting on guesses inside a forty-commit port is how a port
ships a defect that was supposedly already caught.

### 3.4 Apply the findings as a fix wave

- Every finding gets a fix claim. A fix claim is one line naming the exact check that proves the
  fix landed: a grep, a command, a count. If you cannot write the check, the fix is not yet
  specified.
- Run the claim checklist before the wave to get the baseline. All claims failing is the expected
  baseline, and it proves the checklist can fail.
- Apply the fixes to the plan text only. The plan revision must not touch the tree.
- Re-run the checklist after the wave. The delta is the evidence that the fixes landed.
- Keep the findings sidecar next to the plan. It is the input to the next round.
- Record what you refused to fix and why.

A claim checklist is a plain script. One assertion per finding, each one re-runnable later:

```bash
# fix-claims checklist: one assertion per plan-review finding
plan=docs/superpowers/plans/2026-10-18-v0.32.0z-port.md

grep -q 'self._full_attention_groups: tuple' "$plan" && echo "PASS finding-01" || echo "FAIL finding-01"
grep -q 'check 39 40' tools/foldcheck.sh && echo "PASS finding-02" || echo "FAIL finding-02"
grep -q 'expected-red' "$plan" && echo "PASS finding-03" || echo "FAIL finding-03"
```

**Why:** A fix wave is many small edits. Without a per-fix check you are trusting your own memory
of a large diff, and that memory is the thing that just proved unreliable.

### 3.5 Stop when the returns stop

Write the stopping rule into the plan before the first round.

- A round dominated by findings introduced by your own previous fix wave earns exactly one scoped
  pass over the regions that wave changed, then stop.
- Scoped means the changed regions, checked against the findings file, looking for regressions the
  fix introduced. Not six fresh lenses.
- Two rounds is the floor. One is not enough: the second round found high severity defects that
  the first round's revision had introduced itself.

**Why:** Review rounds cost proportionally to the size of the branch, and a review whose output is
mostly your own regressions is buying nothing.

---

## 4. Reviewing each task as it lands

### 4.1 Fresh reviewer per task

- The reviewer is not the implementer and does not inherit the implementer's report.
- The reviewer re-runs the task's acceptance gate itself. "The implementer says it passes" is not
  evidence.
- Give the reviewer the task text, the base SHA, the tip SHA, and the acceptance command. Nothing
  else.
- Run the heaviest task's review on the strongest reviewer available. The task that touches the
  most delicate numeric path deserves more scepticism, not less.

**Why:** An implementer checking its own work reproduces its own blind spots with confidence. The
gate is cheap to re-run and expensive to be wrong about.

### 4.2 Reproduce the red proof

- If a task claims a bug was reproduced before the fix, the reviewer reproduces it. The red proof
  is part of the acceptance, not commentary.
- Prefer a red proof that does not edit the tree. A `sitecustomize` shim that swaps a symbol at
  import time reproduced a liveness failure without touching the working tree.
- If the intended mechanism cannot redden the test, use the stronger proof. Reverting to the
  pre-fix code and showing the test fail is stronger than a swap that cannot produce the failure.
- Run a negative control. Prove the assertion fires because of the behaviour under test and not
  because of how the test is constructed. The same construction without the corruption passes, so
  the assertion is aimed at the right thing.

**Why:** A red proof weaker than the claim lets a green test certify nothing. The negative control
is what separates "the test detects the bug" from "the test errors during construction".

### 4.3 Ledger the rulings, do not absorb them

- Implementer concerns get ruled on in the ledger, in writing, with the fact that settled it.
- Typical rulings: a task's grep gate is dead on arrival because the pattern also matches an
  upstream symbol of the same name, so the grep needs scoping. A pattern is too loose and matches
  two sites where one was intended. A claimed count gate was correct but the plan prose
  contradicted the helper script, so the prose was wrong and the script was kept.
- Record the reason a gate cannot pass at a task boundary and where it will be discharged instead.
  An acceptance surface that cannot go green in this port gets named as expected-red with the
  reason attached, and the real acceptance surface gets named alongside it.
- When a plan premise turns out to be wrong, rule on it explicitly and correct the plan text in
  both places the premise appears.

**Why:** The ledger is the only channel that survives between tasks. A ruling made in conversation
is lost, and the next implementer re-litigates it from scratch.

### 4.4 Verify the history, not just the tree

- Check commit shape per task: expected count, expected subject, expected trailers.
- When a commit message is repaired, verify that every tree hash is unchanged. A message-only
  rewrite must move no content, and the hashes prove it.
- Check trailer placement. A paragraph appended below the trailer block stops the trailer parsing,
  which breaks attribution and compliance. The repair moves the paragraph above the trailer block.
  Where the original commits never had trailers, surface the count and leave them alone rather
  than adding sign-offs that were never there.

---

## 5. Reviewing the finished branch

### 5.1 The branch review is a different review

Per-task reviews check each task in isolation, so they systematically miss defects that only exist
between tasks: asymmetric guards where one side of a new condition reads state the other side
does not, doubled locks, two tasks owning the same file, a gate that passes only because the
previous task had not yet landed its half.

- Run the branch review over the whole branch diff, with the plan-review lens set plus lenses for
  callers and tests.
- Acceptance criterion: behavioural equivalence with the source branch on the new base except the
  ledgered deviations, zero regressions against the stock baseline, and safe to run as the
  production image.
- Seed it with the seams again, plus the acceptance surface and the packaging files.

### 5.2 One verifier per finding

- Dispatch one verification per finding, in waves sized to the concurrency you actually have.
- Each verifier returns confirmed, refuted, or uncertain, with a severity of its own.
- Synthesis merges the verified set. Then a judging pass weighs the findings against evidence the
  diff cannot show: the ledgered deviations, the live boot proofs, the production history of the
  same code path.
- A finding the judge refutes by mechanism stays refuted, and the report says so explicitly along
  with the reasoning.

**Why:** A lens sees only the diff. A judge sees the session. A coverage gap that looks medium in
the diff can be answered by a live proof, and a lint item can be out of scope for a
message-preserving port. Only the judge can tell those apart.

### 5.3 The fix loop

- Elevate only the findings the judging pass confirms. Most mediums stay mediums.
- Fix them on the branch in their own commits, then re-verify.
- When a finding is consciously deferred, note the residual and where it is tracked.
- After any fix that touches a review-sensitive path, re-run the affected lens rather than
  assuming the fix is confined to the finding.

---

## 6. What a review cannot tell you

- A green suite on contended hardware proves less than it looks. Every failure on a
  busy card is an out-of-memory failure until proven otherwise. The fix is to re-run on a free
  card and show the failure disappears, and to record the difference in the ledger.
- A diff review cannot see runtime ordering that depends on undocumented initialisation order.
  That is a design question, not a review finding, and it needs a live boot.
- A review cannot prove the absence of a defect. It can only raise or lower confidence.

---

## 7. Completion gates

- Every plan task is complete and its gate passed.
- Every fix claim from the final wave passes.
- The branch review has no open critical or high findings.
- Every deferred minor is in the ledger with its owner.
- Every owed runtime gate has been discharged, or is carried with a stated reason.
- The branch matches the deviation contract.
- The plan file is in the repository or in its backup, and the ledger is in the worktree.
- Nothing has been published that was not authorised.

---

## Worked examples

One line each.

- The plan claimed a fold was safe because no intervening commit touched the same files. A helper
  script showed two of the folded commits did overlap one later commit at file level. The split
  was still safe, but only because it was disjoint hunk by hunk, and the plan had never said so.
- A review found a plan step that named a symbol the plan itself had said was dead code. That one
  line would have invited the implementer to invent a function and add a call site that never
  existed upstream.
- The strongest acceptance gate in a port was not a test at all. It was a count of collected test
  names before and after a test file merge, which turned a silent deletion of thirteen upstream
  regression tests into a red line.
- A branch review reported an uncaught exception on a cascade path as critical. Two verifiers
  split. The mechanism settled it: the raise needed a key pinned and absent at the same instant,
  and the code path held both conditions in a single synchronous call. The finding was refuted and
  the report said so.
