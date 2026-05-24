#!/usr/bin/env bash
# Drives tests/simulate_p2p_money_flow.py against local pg + redis.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

export DATABASE_URL="postgresql://nrs:nrs@127.0.0.1:55432/nrs_vlt"
export VLT_PG_HOST=127.0.0.1
export VLT_PG_PORT=55432
export VLT_PG_USER=nrs
export VLT_PG_PASSWORD=nrs
export VLT_PG_DB=nrs_vlt
export REDIS_HOST=127.0.0.1
export REDIS_PORT=56379
export REDIS_PASSWORD=
export STRIPE_SECRET_KEY=sk_test_simulated_no_outbound_calls
export JWT_SECRET=sim-jwt-secret-do-not-use-in-prod
export JWT_ALGORITHM=HS256
export POVI_COORDINATOR_SIGNING_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
export POVI_VALIDATOR_SIGNING_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
export POVI_VALIDATOR_ID=sim-validator-1
export POVI_VALIDATOR_KEY_ID=sim-validator-1-key-1
export POVI_COORDINATOR_ID=sim-coordinator
export POVI_COORDINATOR_KEY_ID=sim-key-1
export NRSIP_REGION=local
export POVI_AUTO_ATTACH_MODES=DETERMINISTIC,HYBRID,CREATIVE
export POVI_HIGH_RISK_DOMAINS=finance,medical,legal
export POVI_MIN_VALIDATORS=1
export POVI_DEADLINE_SECONDS=10
export POVI_RECEIPT_TTL_SECONDS=600
export PG_POOL_MIN=1
export PG_POOL_MAX=5
export NRS_ENVIRONMENT=test

cd "$ROOT"
exec /tmp/nrs_sim_venv/bin/python tests/simulate_p2p_money_flow.py
