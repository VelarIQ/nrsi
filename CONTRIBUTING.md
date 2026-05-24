# Contributing

Thanks for helping improve NRSI.

## Scope

This repository is focused on the NRSI runtime/tooling surface. Protocol
specification work belongs in the separate `nrsip-spec` repository.

## Before you start

1. Read `LICENSE`, `PATENTS.md`, and `TRADEMARKS.md`.
2. Open an issue for non-trivial changes.
3. Confirm your change does not include confidential or sensitive data.

## Development workflow

1. Fork and create a feature branch.
2. Keep changes small and focused.
3. Add or update tests for behavior changes.
4. Run relevant test suites locally before opening a PR.

Recommended baseline:

- `python -m pytest global-rollout/tests/test_nrsi_lang.py -v`

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
- compatibility impact for NRSI consumers and SDK integrations.

## Security issues

Do not open public issues for potential vulnerabilities. Follow `SECURITY.md`.
