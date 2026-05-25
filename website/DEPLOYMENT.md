# NRSI Website Deployment Notes

Primary hosting is Google Cloud Run in a dedicated project.

- repository: `VelarIQ/nrsi`
- app source: `web-next/`
- container build: `web-next/Dockerfile`
- deploy script: `scripts/deploy_gcp.sh`

## Primary domain

- `nrsi.ai` (apex)

## Production project

- project id: `nrsi-web-prod-20260524`
- region: `us-central1`
- service: `nrsi-site`

## Deploy to GCP

Example:

```bash
./scripts/deploy_gcp.sh nrsi-web-prod-20260524 us-central1 nrsi-site
```

Then map domain:

```bash
gcloud beta run domain-mappings create \
  --project nrsi-web-prod-20260524 \
  --region us-central1 \
  --service nrsi-site \
  --domain nrsi.ai
```

The command outputs DNS records you must set in Porkbun.

## DNS (Porkbun) for Cloud Run

1. Remove parking records (`pixie.porkbun.com` aliases/CNAME).
2. Add the exact records printed by:

```bash
gcloud beta run domain-mappings describe \
  --project nrsi-web-prod-20260524 \
  --region us-central1 \
  --domain nrsi.ai
```

## CI/CD deployment identity

Cloud Run deploys from GitHub Actions use Workload Identity Federation instead
of static JSON keys.

- workload identity provider:
  `projects/924270273440/locations/global/workloadIdentityPools/github-actions-pool/providers/nrsi-repo-provider`
- deploy service account:
  `nrsi-github-deployer@nrsi-web-prod-20260524.iam.gserviceaccount.com`
- workflow: `.github/workflows/deploy-cloud-run.yml`

You can (re)provision this setup with:

```bash
./scripts/setup_github_wif.sh nrsi-web-prod-20260524 VelarIQ/nrsi
```

GitHub Pages remains available via `.github/workflows/pages.yml` as a manual
fallback only.

## Monitoring and alerts

Provision baseline uptime, 5xx, and latency alerts:

```bash
./scripts/setup_monitoring_alerts.sh nrsi-web-prod-20260524 nrsi.ai nrsi-site
```

Route those alerts through Resend using the alert relay:

```bash
export RESEND_API_KEY='re_xxx'
export RESEND_FROM='NRSI Alerts <alerts@yourdomain.com>'
export ALERT_RECIPIENTS='ops@velariq.ai,leighton@velariq.ai'
export ALERT_WEBHOOK_TOKEN='replace-with-random-token'

./scripts/deploy_resend_alert_relay.sh nrsi-web-prod-20260524 us-central1 nrsi-alert-relay
./scripts/setup_monitoring_resend_channel.sh \
  nrsi-web-prod-20260524 \
  https://nrsi-alert-relay-mp7vgkygjq-uc.a.run.app/monitoring-alert \
  nrsi-resend-alerts-webhook \
  "${ALERT_WEBHOOK_TOKEN}"
```

## Website SSO configuration

The login page now supports Google Sign-In (Google Identity Services):

1. Create a Google OAuth Web Client in project `nrsi-web-prod-20260524`.
2. Add `https://nrsi.ai` to Authorized JavaScript origins.
3. Set the client ID in `web-next/app/login/page.jsx`.
4. Keep `allowedWorkspaceDomain` set to `velariq.ai` to enforce Workspace-only sign-in.

The login flow is SSO-only. Manual username/password fallback is disabled.

3. Optional `www` handling:

- map `www.nrsi.ai` as a second Cloud Run domain mapping, or
- redirect `www` to `https://nrsi.ai`.

## GitHub Pages fallback

- Workflow `.github/workflows/pages.yml` remains available as backup publishing
  path.
- Do not attach `nrsi.ai` to Pages once Cloud Run domain mapping is active.

## Verify

1. `https://nrsi.ai`
2. `https://www.nrsi.ai` (if configured)
3. Confirm managed certificate is provisioned and HTTPS is valid.
4. Check alert policies:
   - `nrsi-ai-uptime-alert`
   - `nrsi-site-5xx-alert`
   - `nrsi-site-latency-alert`
