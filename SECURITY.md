# Security Policy

## Supported release posture

This repository is the public NRSI runtime/tooling repository.

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues.

- Email: security@velariq.com
- Fallback: legal@velariq.com

Include:

- affected component/path,
- impact and attack scenario,
- reproduction steps or proof of concept,
- suggested mitigation (if available).

## Response targets

- Initial acknowledgement: within 3 business days.
- Triage decision: within 7 business days.
- Fix timeline: based on severity and exploitability.

## Disclosure policy

- We prefer coordinated disclosure.
- We will notify reporters when a fix is available.
- Public advisories should avoid exposing unpatched exploit details.

## Scope reminders for contributors

- Never commit `.env` files, credentials, tokens, or private keys.
- Avoid including internal endpoint/account information in docs.
- Run tests and checks before PR submission.

## Related repositories

- NRSI runtime/tooling security issues: report here via the private channel
  above.
- NRSIP protocol-spec issues should be reported in the `nrsip-spec` repository
  using the same private reporting channel.
