# NRSI Website Deployment Notes

Primary hosting is now Google Cloud Run.

- repository: `VelarIQ/nrsi`
- static source: `website/`
- container build: `website/Dockerfile`
- deploy script: `scripts/deploy_gcp.sh`

## Primary domain

- `nrsi.ai` (apex)

## Deploy to GCP

Example:

```bash
./scripts/deploy_gcp.sh prism-production-v2 us-central1 nrsi-site
```

Then map domain:

```bash
gcloud beta run domain-mappings create \
  --project prism-production-v2 \
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
  --project prism-production-v2 \
  --region us-central1 \
  --domain nrsi.ai
```

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
