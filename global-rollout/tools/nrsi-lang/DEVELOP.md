# nrsi-lang development setup

The canonical source files live in `../../nrsi/lang/` (relative to this directory when `nrsi-lang` sits under `global-rollout/tools/`).

For development, install in editable mode:

```bash
pip install -e .
```

The `src/nrsi/lang/` directory should mirror `../../nrsi/lang/`.
To sync:

```bash
rsync -av --exclude='__pycache__' ../../nrsi/lang/ src/nrsi/lang/
```

Or create a symlink (from this `tools/nrsi-lang` directory, target is relative to `src/nrsi/`):

```bash
rm -rf src/nrsi/lang
ln -sf ../../../../nrsi/lang src/nrsi/lang
```

The `../../../../` prefix reaches `global-rollout/` from `src/nrsi/`, then `nrsi/lang` is the canonical tree.
