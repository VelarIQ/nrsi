# Changelog

All notable changes to this repository's open-release baseline are documented
here.

## [0.1.0] - 2026-05-24

### Added

- Apache-2.0 root `LICENSE`.
- `PATENTS.md` and `TRADEMARKS.md` governance policies.
- Community/governance baseline:
  - `README.md`
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
  - `GOVERNANCE.md`
  - `MAINTAINERS.md`
  - `SUPPORT.md`
  - `ROADMAP.md`
  - `COMPATIBILITY.md`

## [0.1.1] - 2026-05-24

### Changed

- Corrected `README.md` quickstart/test commands to use actual repository paths
  under `global-rollout/`.
- Added explicit repository layout guidance for external builders.
- Updated contribution/security language to reflect standalone `nrsi` scope.

### Added

- Repo-native GitHub templates and CI workflows under `.github/`.
- `BUSINESS_MODEL.md` documenting day-one commercialization aligned to ARR and
  funding milestones.

## [0.1.2] - 2026-05-24

### Added

- NRSI website source under `website/` with pricing, business model, enterprise
  CTA, and community resource sections.
- GitHub Pages deployment workflow at `.github/workflows/pages.yml`.

### Changed

- `README.md` now documents website source, deployment workflow, and expected
  Pages URL.

## [0.1.3] - 2026-05-24

### Changed

- Upgraded website from single-page splash to multi-page launch site:
  `index.html`, `pricing.html`, `developers.html`, `enterprise.html`.
- Improved UX/UI styling, responsive navigation, and reusable client script.

### Added

- `website/CNAME` set to `nrsi.ai`.
- `website/DEPLOYMENT.md` with Porkbun DNS + GitHub Pages setup instructions.
- Pages workflow environment updated to force Node 24 JavaScript actions.

## [0.1.4] - 2026-05-24

### Added

- Cloud Run deployment artifacts for website hosting:
  - `website/Dockerfile`
  - `website/nginx.conf.template`
  - `website/docker-entrypoint.sh`
  - `scripts/deploy_gcp.sh`

### Changed

- Website deployment guidance now prioritizes GCP Cloud Run with `nrsi.ai`.
- Removed Pages `CNAME` file to avoid domain ownership conflicts during Cloud
  Run cutover.

## [0.1.5] - 2026-05-24

### Changed

- Expanded website UX/UI to a fuller information architecture with platform and
  contact flows integrated across navigation.

### Added

- New website pages:
  - `website/platform.html`
  - `website/contact.html`
- Search/discovery assets:
  - `website/sitemap.xml`
  - `website/robots.txt`
