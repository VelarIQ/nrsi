# NRSI Website Deployment Notes

This static site deploys via GitHub Pages from:

- repository: `VelarIQ/nrsi`
- source artifact path: `website/`
- workflow: `.github/workflows/pages.yml`

## Primary domain

- `nrsi.ai` (apex)

## DNS (Porkbun) for GitHub Pages

Remove `pixie.porkbun.com` parking records and set:

### Apex (`@` / `nrsi.ai`)

- A -> `185.199.108.153`
- A -> `185.199.109.153`
- A -> `185.199.110.153`
- A -> `185.199.111.153`

### WWW

- CNAME `www` -> `velariq.github.io`

Optional:

- URL redirect `nrsilang.dev` -> `https://nrsi.ai/developers.html`

## GitHub Pages settings

- Build/deploy source: GitHub Actions
- Custom domain: `nrsi.ai`
- Enforce HTTPS: enabled

## Verify

1. `https://nrsi.ai`
2. `https://www.nrsi.ai`
3. Confirm HTTPS certificate is valid and auto-renewed.
