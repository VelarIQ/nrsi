"""Claude.ai parity battery scripts.

These scripts run live HTTP probes against a local NRS stack
(``platform-api`` + ``nrs-worker`` + ``edge-gateway``) and verify that
every response from /v1/chat/completions and /peers/* meets the
quality bar described in CLAUDE-PARITY.md.

CI invokes them via ``run_all.py`` which boots the docker-compose
stack, waits for healthchecks, mints a JWT, then runs each battery in
sequence. Locally you can run individual batteries with the same
NRS_BASE / NRS_API_KEY env vars they expect.
"""
