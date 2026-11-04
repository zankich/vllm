# Release and Pitfalls

Release mechanics that repeat every port: cutting the release branch, building the image from a pushed-tip diff, tagging and pushing, the release ledger, and recurring pitfalls checked by name.

---

## 0. Preconditions

- The port branch is green: every commit landed green, the deviation contract holds, and the
  branch review has passed.
- The branch is pushed, and the tree that will be built is the tree that was reviewed. A build
  from a dirty working tree publishes something nobody looked at.
- The upstream base image for the release tag exists in the registry. The upstream image for a
  release tag appeared roughly 34 hours after the tag in one case. Check rather than assume.
- The upstream base is the release, not a release candidate and not a pinned nightly. A nightly
  base changes the patch base, and every artifact in the port becomes unreproducible.
- You have the previous release's published tag and its digest recorded, so rollback is a pin
  change rather than a rebuild.

---

## 1. Cut the release branch

- **Name the branch `vX.Y.Zz`, matching the upstream release tag.** The `z` suffix is the fork
  marker and it sorts the fork's branches next to the upstream release they sit on.

- **Drop the model-specific suffix once the fork carries more than one model's changes.** A fork
  that started as one model's fork and grew into a general-purpose fork keeps the narrow name in
  its tag, its README, and its image tag until someone renames it. The rename is a release, not
  a cleanup: the branch name, the image tag, the README headings, and the plugin's own
  identification all have to move together.

- **Cut the worktree from the upstream release tag, not from the previous fork branch.** The
  previous fork branch is the thing being replaced, and it carries the drift you are about to
  map.

- **Do not rebase onto a release candidate.** Wait for the release cut. A branch rebased onto an
  rc that later changes is a branch that has to be rebased again, and the second rebase is where
  the conflicts get expensive.

- **Map the drift before writing a single line of the plan.** The previous release's plan is the
  structural template. Reuse its section shape, not its content.

---

## 2. Pre-release hygiene: diff baseline and image consistency

The image is built from a diff, not from the tree. Everything in this section exists to make the
diff an accurate, reproducible description of the fork.

- **Generate the artifact diff from the pushed remote tip, never from the local working tree.**
  `git diff <upstream-tag>..origin/<branch> -- <image paths>`. Confirm the local tree has nothing
  uncommitted that would be missing from the artifact, and confirm the artifact is generated
  inside the build, not checked in by hand.

- **Scope the pathspec to the paths the image actually contains.** The diff is applied inside the
  image with `patch -p1`, against the base image's installed tree, which does not have the
  repository's build tooling in it. A diff that reaches outside the image's paths fails to apply
  at build time, and one that stops short of the image's paths leaves the fork's code out of the
  artifact entirely.

- **The diff carries new files.** A file the fork adds appears in the diff as an addition against
  `/dev/null`, and it lands in the image the same way as a modification. This is how the plugin
  and the out-of-tree modules get into the image. They do not need a separate copy step.

- **Overlay the patch onto the upstream base. Never copy the fork tree into the image.** A copy
  step pulls in `.git`, untracked files, and local build residue, and silently ships whatever was
  in the working directory. The patch approach also leaves the base image's own CUDA-aware C
  extensions intact, which a full tree copy does not.

- **Install plugins with `pip install <dir>`, not `COPY`.** Copying a plugin directory into the
  image skips the entry-point registration, and the plugin then imports without registering.

- **Verify the image's contents inside the image, never against the source tree.** The source
  tree can be clean while the image layer carries a stale copy of the plugin, because the
  plugin layer can be built from a copy that drifts behind the canonical one. A verification pass that greps the source tree proves nothing
  about the image. Grep the image.

- **The `vllm[audio]` extras, and the resolver conflict behind it.** The upstream wheel's
  metadata pins a NCCL version that disagrees with the NCCL the base image ships, because the
  base image installs its own newer NCCL. Any `pip install "vllm[...]"` triggers a
  dependency-consistency pass that "fixes" the disagreement: unpinned, it silently downgrades
  NCCL, which changes the collectives the engine actually runs on. Pinned in the same command,
  the resolver backtracks to an ancient `vllm` release from PyPI whose metadata happens to
  satisfy the pin, and it replaces the patched tree wholesale.

- **Install the extras and restore NCCL as two separate invocations.** One resolver pass asked
  for both backtracks and replaced the tree. Two passes do not. Name the extras directly and let
  upstream own the dependency list, then restore the base image's NCCL in a separate invocation.

- **Add a drift check to the build.** `pip show vllm` in the verification layer, asserting the
  installed distribution is the fork's own and not a PyPI replacement, turns a silent tree
  replacement into a failed build. This check is the only thing that stands between a resolver
  backtrack and a published image with the wrong code in it.

- **Call the plugin's install check inside the verification layer.** A Dockerfile that only
  greps for source text will pass a build whose base tag has drifted underneath a pinned
  plugin, and the failure then surfaces at the first serve boot instead of at the build.

- **Expect the layer cache to lie.** The patch layer hits only when the diff output is
  byte-identical, and a registry digest commits to the exact layer set, so floating dependency
  versions change the digest even when the application code is unchanged.

- **Match the interpreter the base image ships.** The vLLM images provide `python3` only, so a
  `RUN python -c ...` sanity line exits 127 and fails the build for a reason that has nothing to
  do with the code. Use the same interpreter in any healthcheck.

---

## 3. Image publishing: tagging, pushing, and digest pinning

- **Build under an rc tag locally, always.** A published tag must not be able to move because
  someone ran a build.

- **Push the branch first, then the tags, then the image**, in that order, and check the remote
  listing before and after each outward-facing change.

- **Stamp the versioned tag and the moving tag on a canonical build. Do not stamp the moving tag
  on an explicitly named build.** The default path produces `<version>` and the moving tag
  together. A build invoked with an explicit tag, an rc, an experiment, a local-only variant , 
  produces only that tag. A build that stamps the moving tag as a side effect promotes every
  stack that references it on the next recreate, and an experiment build that did that is a
  production change nobody approved.

- **Keep the push separate from the build.** The script tags and builds. Pushing is a deliberate
  step. This is what makes "build and publish" two actions instead of one, and it is what lets
  the artifact be inspected before it becomes public.

- **Do the moving-tag promotion and the default-branch flip as two separate, separately
  authorised actions.** Both are outward-facing and both change what downstream consumers pull. A request to
  publish a version is not a request to make it the default, and the two have been asked for
  together and separately.

- **A local moving tag can be overwritten by an ordinary build.** Restore it from the registry
  before pushing, so a local rebuild never silently demotes the published tag.

- **Inspect the registry manifest after pushing.** Push output is not proof that the tag you
  wanted points where you think it points, and it is not proof that the other tags are still
  where they were.

- **Pin from the registry's actual tag list, not from the package version string.** The image
  tag format and the pip version format differ, and when one token is expected to drive both
  forms the tag list is the authority.

- **Record the digest beside every benchmark result.** A restart against a mutable tag is no
  longer bit-reproducible, so a number measured against one digest is meaningless against
  another.

- **Expect the digest to differ between two builds of the same commit.** Re-running the pip
  layers can float a dependency version and change the layer set. Compare the application layers
  when the digests differ and say plainly whether the content is equivalent. Do not claim the
  digests are equal when they are not.

- **Confirm what a moving-tag promotion does to every consumer before you push it.** Anything
  referencing the moving tag with an always-pull policy picks the new image up on its next
  recreate. Announce that consequence as part of the push, not after it.

---

## 4. Deferred items and the release ledger

- **A deferral is a decision when it is written down, and a defect when it is not.** Every
  deferred item carries a name, a reason, an owner, and the design that would close it. A fix
  shape that has been proposed and ruled on by the owner is recorded as ruled, not as pending.
  A boot that has been deferred carries the plan constraint it implies, so the deferral also
  names what must not ship until it runs.

- **Keep the deferred list out of the public README.** A README that lists owed work is a
  roadmap. The operational reference should describe the artifact as it is.

- **The default-branch flip is its own item, not part of the publish.** It is recorded as an
  open decision until it is made, and made only on request.

- **Outstanding administrative items are tracked explicitly, not by memory.** The cache cleanup
  that needs elevated privileges, an infrastructure fix, an uncommitted configuration change,
  the legacy mirror tag that still points at a superseded digest, and the blocked checkpoint
  restoration are each a line in the ledger with a name and an owner. A release that closes with
  no open ledger lines closes cleanly. A release that closes with five undocumented ones does
  not.

- **The ledger is not closed by the publish.** Publication closes the artifact. The ledger closes
  the release.

---

## 5. Recurring pitfalls

These are the traps that have appeared more than once. They are listed here because each has
cost a release cycle, and each will cost another one if it is not checked by name.

- **The artifact diff generated from the working tree instead of the pushed tip.** The image
  carries a commit that was never pushed and is not in the branch anyone reviewed.
- **The resolver conflict between the extras and the base image's NCCL.** Unpinned, it silently
  downgrades NCCL. Pinned in the same command, it backtracks to a PyPI `vllm` and replaces the
  patched tree.
- **The missing drift check.** Without `pip show vllm` in the verification layer, the tree
  replacement above is invisible until the served behaviour is wrong.
- **The local moving tag overwritten by an ordinary build.** The published moving tag is
  demoted without any push happening.
- **The experiment build that stamped the moving tag.** Every stack on an always-pull policy
  picks up an experiment on its next recreate.
- **The digest that changed with no code change**, and the response of "the content is identical,
  never mind". Say what was compared, and say what was not.
- **The force-push that failed on a stale lease.** Use an explicit expected-value lease rather
  than a blind force, and check the remote state first.
- **The `.gitignore` rewritten by sorting**, which produces a large diff against the upstream
  ordering and buries the one line that changed. Edit in place.
- **The unbounded history rewrite**, which walked into a 2019-era upstream comment and rewrote
  upstream ancestry. Bound the range.
- **The commit whose existence reveals the purge.** A commit subject containing the scrub's own
  vocabulary tells a reader that something was removed. The rewritten history should read as
  though the leaks never existed.
- **The health check whose pattern matches a substring of the failure it is looking for**, so a
  booting engine reads as a dead one.
---

## 6. Release gates

Before the branch is published and the image is promoted, all of these must be true and recorded:

- The release branch is cut from the upstream release tag, pushed, and green.
- The tree that was built matches the tree that was reviewed, and the working tree is clean.
- The artifact diff is generated from the pushed tip and is scoped to the image paths.
- The verification layer in the built image is green, including the plugin pin check and the
  package drift check.
- The image content was verified inside the image, not in the source tree.
- The registry manifest shows the intended tags, and the tags that were not meant to move are
  still where they were.
- The moving tag was promoted deliberately, and the default branch was changed only if it was
  separately asked for.
- The published digest is recorded beside every measurement taken against it.
- The README and every commit message make no claim a gate did not support, and leak no
  hostnames, addresses, personal details, fleet topology, or internal process language.
- The release ledger is closed, or every open line has a name, a reason, and an owner.

---

## Worked examples

Each of these cost a release cycle. They are here as concrete instances of the bullets above, not
as a narrative.

- The image that carried every scrubbed string because the plugin layer was built from a copy
  that predated the entire scrub campaign, and the verification that had been grepping the
  source tree instead of the image.
- The build that downgraded NCCL from 2.30.7 to 2.29.7 while resolving the audio extras, and
  the pinned build that went further and replaced the patched tree with an ancient release from
  PyPI because the resolver backtracked to satisfy the pin.
- The drift check that caught it, which is the only reason either was found before publication.
- The published moving tag demoted by an unrelated local rebuild.
- The `git diff` generated from the working tree, and the commit that was in the image but not on
  the branch.
- The force-push that needed an explicit lease because the local lease information was stale.
- The history rewrite that walked into a 2019-era upstream comment and diverged the whole
  ancestry, and the range-limited rewrite that did not.
