# Release Process

Use tag-driven GitHub Actions releases to build, validate, create a GitHub Release, and publish to PyPI.

Workflow file: `.github/workflows/release.yml`

## Standard Release Flow

1. Ensure `main` is green.
2. Update `CHANGELOG.md`.
3. Bump the version:

```bash
python tools/bump_version.py 0.1.1
```

4. Commit the version/changelog update.
5. Create and push an annotated tag:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

The release workflow validates that:

- The tag is semantic version format: `vX.Y.Z`
- The tag matches `project.version` in `pyproject.toml`
- `pyproject.toml` matches `src/floudsonnx/__init__.py`
- The package builds successfully
- `twine check dist/*` passes

## PyPI Trusted Publisher Setup

In PyPI, add a trusted publisher with:

- Owner: `gmalakar`
- Repository name: `floudsonnx`
- Workflow filename: `release.yml`
- Environment name: `flouds_by_goutam`

The environment name must match the `publish-pypi` job in `.github/workflows/release.yml`.

## Manual Re-Release

Use Actions -> Release -> Run workflow for an existing tag.

- Set `tag` to the existing tag, for example `v0.1.1`.
- Set `publish_to_pypi=true` only if that version has not already been uploaded to PyPI.

PyPI rejects duplicate uploads for an existing version.
