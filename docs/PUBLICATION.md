# First public research snapshot

Publication authorized on 2026-11-03. Development baseline: `10aeddf`, implementation validation: `f849eae`. The latest local branch was checked and contained no newer commits or uncommitted work at the start of publication preparation.

The public repository starts with a clean snapshot; development commit objects contain personal author email metadata and are retained locally. No development branch is rewritten. Original commit IDs remain provenance identifiers and may not be resolvable in public Git history.

The publication includes Python source, example configurations, tests, documentation, a continuation-state JSON and `public_results/v20-summary.json`. A pattern scan of 113 historical blobs found no private-key headers, common credential formats, private user-home paths or email addresses in file content; no historical blob exceeded 1 MiB. Commit metadata was audited separately. This is a bounded pattern scan/manual review, not a guarantee of absence of all possible secrets.

Ignored model weights, raw datasets, GPU traces, raw `results/`, local environments and publication staging files are excluded. The public summary selects numeric measurements and SHA256 associations only; raw reports and environment dumps with machine paths are not uploaded. Example cross-project paths use sibling repositories. A pristine source-only installation can execute the synthetic CPU pipeline; real-model experiments require authorized external artifacts matching the published hashes.

The first publication does not add a new license grant or distribute external code/data/weights. A licensing decision is still needed before representing the repository as licensed open source. V2.0 remains in progress, with limitations documented in README and the implementation status.

Git transport could not connect directly to GitHub. Publication uses the connected GitHub API, followed by remote commit/tree verification. No force push or historical rewrite is used.
