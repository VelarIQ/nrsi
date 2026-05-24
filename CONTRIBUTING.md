# Contributing

Thanks for helping improve NRSI and NRSIP.

## Scope

This monorepo is in transition toward two public repositories:

- `nrsi` for runtime/tooling.
- `nrsip-spec` for protocol specification and reference assets.

Contributions should clearly indicate which surface they affect.

## Before you start

1. Read `LICENSE`, `PATENTS.md`, and `TRADEMARKS.md`.
2. Open an issue for non-trivial changes.
3. Confirm your change does not include confidential or sensitive data.

## Development workflow

1. Fork and create a feature branch.
2. Keep changes small and focused.
3. Add or update tests for behavior changes.
4. Run relevant test suites locally before opening a PR.

## Commit and PR expectations

- Use clear commit messages describing intent.
- Include a concise PR description:
  - problem statement,
  - solution approach,
  - test evidence,
  - risk/rollback notes.
- Link related issues.

## DCO sign-off

By submitting a patch, you certify you have the right to submit it under
Apache-2.0. Please add a Signed-off-by line in commits:

`Signed-off-by: Your Name <you@example.com>`

## Code review criteria

PRs are reviewed for:

- correctness and regression risk,
- security and confidentiality impact,
- test coverage and maintainability,
- compatibility impact for NRSIP consumers.

## Security issues

Do not open public issues for potential vulnerabilities. Follow `SECURITY.md`.
