# CLAUDE.md

AGENTS.md is the single source of truth for this repository: rules, codebase map, invariants,
pitfalls, test placement and release steps. It is imported below so Claude Code loads it
automatically. Update AGENTS.md, not this file, when project knowledge changes.

@AGENTS.md

## Claude Code specifics

- Before declaring a task done, run at least `scripts/verify_release.sh --quick`. For changes that touch
  Linux-only code, IPC, ingest, the fleet controller or S3, run the full harness (Docker stages
  included) and report the summary table faithfully.
- Do not commit, tag or push unless the maintainer explicitly asks. They push releases manually.
- Long harness runs (the `linux` and `image` stages take several minutes) belong in the background.
  Do not edit `scripts/verify_release.sh` while one is running.
- Never "fix" the user's Docker or toolchain setup on their machine. Work around it (for example
  `BUILDX_CONFIG=$(mktemp -d)`) and tell them what you found.
