# Releasing

The package version is sourced from `__version__` in [src/opbdh/__init__.py](src/opbdh/__init__.py); `pyproject.toml` reads it via hatch.

## Recommended: GitHub Release → PyPI Trusted Publishing

One-time setup on PyPI (project **opbdh** → Publishing → add a trusted publisher):

- Owner/repository: this GitHub repo
- Workflow file: `publish.yml`
- Environment: `pypi`

Then, for each release:

1. Bump `__version__` in `src/opbdh/__init__.py` and merge to `main`.
2. Wait for CI to pass.
3. Publish a GitHub Release with a `vX.Y.Z` tag.

[`.github/workflows/publish.yml`](.github/workflows/publish.yml) runs the test suite, builds the sdist and wheel, and uploads to PyPI via OIDC — no password or API token involved.

## Manual fallback

```bash
python -m pip install -U build twine
python -m build
twine check dist/*
```

Upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
```

Then to PyPI:

```bash
twine upload dist/*
```
