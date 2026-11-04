# Verify the Port

Prove the mapped result is what you meant to ship: a real baseline, gates that can fail, an artifact that matches the tree, runtime behavior, quality, and documentation that claims nothing a gate did not support.

---

## 0. Preconditions

- The baseline exists before the first pick lands. A comparison taken after the change is not a
  comparison.
- The baseline was produced by the exact invocation the final validation will use. A baseline from
  a different command line, different flags, or different parallelism is not comparable.
- No two GPU suites run at the same time while a baseline or a gate runs. Parallel suites contend
  for devices and produce failures that read exactly like regressions.
- Nothing has been pushed, no image tag has moved, and the fork default branch is unchanged.
  Verification happens against local state.
- You have a known-good image of the branch you are replacing. Nearly every port question reduces
  to a comparison against it, so it must exist before the first question arrives.

---

## 1. Record the baseline honestly

- Write the baseline command into the plan verbatim, including flags, environment, and paths. The
  final validation runs that same string.
- The baseline command may only name paths that exist on the base tree. A missing path makes the
  whole invocation refuse to run, and the empty result looks like a clean baseline.
- Fork-only test files do not exist on the base tree. Run them after the pick that lands them and
  treat their outcome as new coverage, not as baseline.
- Capture the full failure list yourself. The test cache mixes runs from different sessions, and
  stale entries from an older session make the cached list wrong.
- Decompose the baseline by class rather than by count. A baseline of 49 failures that are all
  device-capability or allocation failures is a different object from 49 failures spread across
  logic, and only the second one constrains you.
- Run the suites serially. A server-boot suite running against a GPU suite has produced phantom
  failures that disappeared when both were re-run one at a time.
- Record the environment with the numbers: package versions, driver, hardware, model, and the
  flags under test. An unexplained difference later will be unresolvable.
- **Why:** a baseline you cannot re-derive is a feeling, not a reference. Its only value is being
  reproducible on demand.
- If the reference measurement turns out to be defective, say so and re-run it before comparing
  anything. A number collected on a broken build is worse than no number, because it looks like a
  threshold.
- Keep the baseline in the worktree ledger next to the plan, not in a scratch directory that gets
  cleaned.

---

## 2. Prove the gate can fail

- Every new test is observed red before its fix lands. A test written and passed in one sitting
  proves nothing about the bug it claims to cover.
- Re-derive the red signal. Revert the fix, run the test again, and confirm it fails on the
  expected assertion rather than on a collection error, a missing import, or a mock that no longer
  binds.
- Confirm the test value exercises the property. A test that passes with the property broken is not
  a gate, and it will silently retire at the next upstream refactor.
- Audit the committed evidence. Read the test file as committed at the fix and confirm it is the
  same body that failed, not a rewritten version that only ever passed.
- Gate on test names and counts as well as on the failure diff. A deleted, renamed, or skipped test
  produces no failure and is a silent regression. Collect the names before and after and diff them.
- Patch the symbol the module actually imports. A mock of a symbol the module no longer imports
  fails loudly instead of testing nothing, which is the good case, but it also means the mock has
  to be retargeted at the new call surface.
- Assert constants and types against the real module rather than against a local copy. A copied
  constant drifts from its source and the assertion keeps passing.
- Exercise the production entry points. A test that calls a private helper proves the helper
  works, not that the server uses it.
- Validate the gate harness against a live response before trusting its verdict. A harness that
  inspects the wrong field will report success on a broken response.
- **Why:** the whole cost model of a port assumes the gates are informative. One vacuous gate
  silently converts every later step into an assertion without evidence.

---

## 3. Gate the code paths

- Fix open-failing harnesses first. A harness that is already red when you arrive will mask your
  own failures and you will not be able to tell them apart.
- Run every suite the port touched, not just the suites the new tests live in.
- Use persistent volumes for destructive tests such as wiping a cache tier mid-flight. A test that
  destroys shared state cannot be re-run without cleaning up after itself.
- Install package extras in a separate step from the tree install, and prove the tree survived:
  take a package location diff before and after the extras install, and fail the build loudly if
  the installed location moved.
- Keep the fail-loud drift contract at import. The check that the patched tree is still the patched
  tree must raise at import time rather than letting the engine serve quietly wrong code.
- Make load-time hooks idempotent. An install hook that runs twice, or that re-wraps after its own
  install, must not be able to double-apply or silently drop itself. Test the second run.
- Assert the upstream refusal you depend on. If the fork relies on upstream rejecting a
  configuration, assert that rejection so an upstream change that removes it fails your suite
  instead of changing behaviour underneath you.
- Densify the observation interval for rare races. A once-per-minute probe cannot see a
  once-per-ten-minutes failure, so shorten the interval and lengthen the soak when hunting one.
- Keep the collection of the failing path honest. If the test needed a stub for a tier it cannot
  build, use a stub that still exercises the wiring contract rather than skipping the case.

---

## 4. Prove the artifact is the artifact

- Regenerate everything from one build script. Two ways to build means the published image can
  come from the one you did not review.
- Verify the tag wiring without a multi-gigabyte build by running the script against a stubbed
  container client, and cover both the default-tag path and the explicit-tag path. The two paths
  have behaved differently before, with an explicit tag incorrectly stamping the moving tag.
- Inspect the registry manifest after pushing. Push output is not proof that the tag you wanted
  points where you think it points.
- Treat a cache hit as proof of byte identity only when the input was byte-identical. A cache hit
  on a changed input is a silent miss.
- Diff the package location inside the built image against the fork tree. This catches an install
  step that quietly swapped the tree for the published one.
- Verify the content of the running container, not just its tag. Confirm the fork code and the
  plugin are present and importable inside the container you are about to serve from.
- Check flag names against the pinned image's own help output. Flag spellings drift between
  releases, and a flag that is silently ignored looks exactly like a flag that worked.
- Verify the remote state with a remote listing before and after any outward-facing change, and
  confirm the default branch and the other published tags are untouched by name.
- Generate the artifact diff from the pushed remote tip, never from the local working tree, and
  confirm the local tree has nothing uncommitted that would be missing from the artifact.
- **Why:** every image is an overlay of a patch onto an upstream base. If the patch
  and the base are out of step, the running code is not the code you reviewed.
- Verify intermediate commits stay green as they land and that the final tree identity holds
  against the tree you validated. A green tip can hide a red commit that was later edited.

---

## 5. Prove the runtime

- Boot the new build against a known-good image of the old build and compare behaviour. One build
  at a time, or the comparison means nothing.
- Verify the deployed configuration matches the fix. Model path, flag set, tier sizes, and cache
  directories are all part of what you are testing. A wrong model path produces a stack that boots
  cleanly and serves the wrong thing.
- Read the resolved values out of the boot log rather than out of the config file. The log states
  what was actually applied, including the mode the engine decided to run in and the dtype it
  actually selected.
- Take the cache capacity from the resolved boot-log numbers. The configured value and the
  effective value differ often enough that tuning from the config alone misleads.
- Gate the tier spill and restore path explicitly. Fill the tier until it spills, then prove the
  restore path returns what was stored.
- Run a cold restart gate. Restart the engine, restore from the disk tier, and compare the served
  output byte for byte against the pre-restart output.
- Soak at full context with a marker in the output so you can tell the soak traffic from the
  ordinary traffic when you read the logs.
- Serialize load runs. Concurrent load runs against the same engine produce a measurement of
  contention rather than of the change.
- **Why:** the failure classes that survive unit tests are the ones that only appear across a
  restart, a spill, or a saturated device. The runtime is where they live.
- Re-run the serving validation after every fix that touched a runtime path. A serving result
  collected before the fix describes the old code.

---

## 6. Prove quality did not move

- Determinism is not correctness. Byte-identical output across two runs of the same build says
  nothing about whether the output is right.
- Scope the determinism claim precisely: same prompt, same build, same configuration. Do not
  present it as a property of the model or of the port.
- Check output quality separately from output stability, with an actual quality measurement.
- Qualify the quality number against the stored baseline and against the noise floor of the
  harness. A change smaller than the noise is not a change.
- Measure the benefit at an identical memory pin. A win that also changes how much memory the
  engine uses is not a like-for-like win.
- Re-run the benchmark when the baseline was defective. Comparing a fresh run to a compromised
  baseline produces a false regression, and the false regression then drives a wrong decision.
- Recalibrate whenever the model or the quant changes. A calibration sidecar measured on one
  checkpoint does not transfer to another, so collect your own on your own shape rather than
  serving someone else's artifact.
- Corroborate a recalibration against an independent reference run before adopting it. Two
  measurements that agree are worth one that agrees with itself.
- Regenerate the calibration artifact from the raw measurements. Editing the output file breaks
  the chain back to the data that justifies it.
- Merge calibration dumps by taking the union maximum per key, so a merge never lowers a ceiling
  that a stricter run established.
- Read the clip counters as a distribution, not as a single line. One element past the ceiling is
  a tail poke, while counts that climb every interval mean the ceiling is in the wrong place and the
  scale needs recalibrating.
- **Why:** a port can preserve every functional gate and still ship a model that answers worse.
  Quality gates catch only the regressions that the functional gates cannot see.

---

## 7. Triage failures before diagnosing them

- Reproduce the failure before you trust any diagnosis, including your own.
- Treat a user correction as a hard rule over your analysis. When the maintainer says a path works
  fine and the comment claiming otherwise is wrong, the comment is what gets removed.
- Change one variable at a time. Two changes at once give you a result you cannot attribute.
- Isolate variables by bisection rather than by intuition when a failure appears only under load.
- Group warnings by request and by time before reading them. A burst of identical warnings across
  many requests is a different thing from the same warning repeated by one request.
- Classify transient saturation separately from corruption. A store that fails while the tier is
  pinned at its ceiling is a capacity symptom, and it resolves when the ceiling moves.
- Do not trust the severity label on a log line. A line logged at error level inside a
  best-effort background thread may be a benign failure, and a line logged at info level may be the
  only evidence of a real problem.
- Correlate the symptom with the deployment date and the logs from the window in which it first
  appeared. Symptoms that predate the port are not caused by the port.
- Match structures one for one when you claim two implementations are equivalent. Count the
  pieces on both sides.
- Instrument the running system when static reading cannot settle a behavioural question. A
  temporary counter answers what code inspection guesses at.
- Escalate content-correctness questions to whoever owns the response contract. Whether an answer
  is acceptable is not a question the port should answer on its own.
- Re-test a blocker under clean conditions before you treat it as a blocker. Device contention,
  leftover state, and orphaned processes all masquerade as real failures.
- Recalibrate an invariant when reality contradicts it. An invariant that fires on valid states
  is a hypothesis that failed, and the right response is to narrow it to the states that are truly
  impossible, then make sure it still fires for those.
- **Why:** most of the wasted port time goes into diagnosing failures the port did not cause, and
  the failure of a diagnosis is cheap to fix only if you catch it early.

---

## 8. Verify the documentation against the truth

- Check every documentation claim against ground truth in the repository, not against your notes
  from the port. Each claim needs a check you can re-run.
- Mark gates that have not run as pending. A document that describes an unproven state as
  achieved is worse than an absent document, because it will be trusted.
- Run the claim check adversarially and accept that it will find wording that overstates the
  evidence. Those catches are the point of the pass.
- Keep hostnames, addresses, personal details, fleet topology, and internal process language out
  of public documentation and out of commit messages.
- Do not leave owed-item language in public documentation. Move it to the private ledger.
- Keep commit messages declarative and short, describing what the diff does rather than how it
  was reached.

---

## 9. The final gate list

Before the branch is pushed and the image is promoted, every one of these must be true and
recorded:

- The baseline exists, uses the final invocation, was captured serially, and its failure classes
  are decomposed.
- Every new test was observed red before its fix, and the red signal was re-derived from the
  reverted code.
- The collected test names and counts match the baseline set plus exactly the intended additions.
- Every suite the port touched has been run, and the failure set is the baseline set plus nothing
  new after environmental failures are separated out.
- The artifact diff comes from the pushed tip, the working tree is clean, and the package location
  inside the built image still points at the fork tree.
- The registry manifest shows the intended tags, and the default branch and the other tags are
  unchanged.
- The new build has booted against a known-good image, produced coherent output, restored from the
  disk tier byte for byte after a cold restart, and soaked at full context.
- The quality measurement sits inside the noise band of the stored baseline, or the deviation is
  explained and accepted.
- The documentation makes no claim that a gate did not support.
- Nothing in the branch, the tags, or the messages leaks hostnames, addresses, personal details, or
  internal process language.

---

## Worked examples, one line each

- **A red gate that was not a gate:** a test mocked a symbol the module had stopped importing, so
  it failed loudly on every run and nobody could tell whether the property was covered. Retargeting
  the mock at the real call surface produced the first honest result.
- **A baseline that lied:** two GPU suites were baselined in parallel and the contention failures
  landed in the baseline. Re-running them serially moved 17 errors down to a stable 7 plus 10, and
  the real red signal only appeared after that.
- **A fix that silenced rather than corrected:** a capacity invariant computed coverage from full
  chunks only and ignored the tokens carried by the boundary key, so every partial-tail restore
  looked like a violation. Narrowing the invariant to truly impossible states kept the signal
  meaningful.
- **A cross-version diff that proved nothing:** the older engine produced different output for the
  same prompt across runs, so the output comparison between versions had no signal. The comparison
  moved to quality evaluation, and the stability check stayed inside one build.
- **An artifact that was not the tree:** a build step installed the published package and pulled
  the fork's tree out of the image. The location diff before and after the install turned a silent
  substitution into a build failure.
