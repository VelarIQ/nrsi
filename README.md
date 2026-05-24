# NRSI

NRSI is the neuromorphic reasoning runtime and tooling surface for the broader
NRS ecosystem.

## License

- Code: Apache-2.0 (`LICENSE`)
- Patent policy: `PATENTS.md`
- Trademarks: `TRADEMARKS.md`

## Repository layout

- `global-rollout/nrsi/`: core NRSI runtime and language implementation.
- `global-rollout/tools/nrsi-lang/`: compiler/toolchain packaging and docs.
- `global-rollout/sdks/`: SDK surfaces for external builders.
- `global-rollout/tests/`: integration and subsystem test suites.

## Quick start

1. Install Python dependencies:
   - `python -m pip install -r global-rollout/requirements.txt`
2. Run a focused validation test:
   - `python -m pytest global-rollout/tests/test_nrsi_lang.py -v`
3. Run broader suite (optional):
   - `python -m pytest global-rollout/tests -v`

## Contributing and support

- Contributing guide: `CONTRIBUTING.md`
- Security reporting: `SECURITY.md`
- Governance model: `GOVERNANCE.md`
- Roadmap: `ROADMAP.md`
- Day-one commercialization model: `BUSINESS_MODEL.md`

## NRSI website

- Source: `website/`
- Primary deployment target: GCP Cloud Run via `scripts/deploy_gcp.sh`
- Domain target: `https://nrsi.ai` (see `website/DEPLOYMENT.md`)
- GitHub Pages workflow: `.github/workflows/pages.yml` (fallback path)
- Site map:
  - `index.html`
  - `platform.html`
  - `pricing.html`
  - `aeo.html`
  - `developers.html`
  - `enterprise.html`
  - `contact.html`
  - `login.html`
  - `dashboard.html`
  - `admin.html`
