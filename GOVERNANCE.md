# Governance

This project is maintained under a maintainer-led model while NRSI and NRSIP
are separated into dedicated public repositories.

## Project surfaces

- `NRSI`: runtime/tooling implementation.
- `NRSIP`: protocol specification and compatibility guidance.

Each surface may evolve on independent release cadences.

## Roles

- Maintainers: review and merge changes, manage releases, and enforce policy.
- Contributors: submit issues, proposals, and pull requests.
- Owners: final decision authority for legal, security, and trademark matters.

## Decision process

1. Open proposal via issue/PR.
2. Gather maintainer review and technical feedback.
3. Reach rough consensus when possible.
4. If blocked, maintainers decide; owners resolve legal/security disputes.

## Compatibility and stability

- Breaking changes require explicit callout in `CHANGELOG.md`.
- Protocol-impacting changes should include compatibility notes in
  `COMPATIBILITY.md`.

## Governance changes

Updates to this file require maintainer approval and a visible rationale in the
associated PR.
