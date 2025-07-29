# Release Guide

This document describes the release process for the CANNs library.

## Version Management System

The CANNs project uses a centralized version management system that ensures consistency across all components:

- **Git tags**: Source of truth for version numbers (format: `v0.5.1`)
- **Package version**: Automatically derived from git tags using `uv-dynamic-versioning`
- **Documentation**: Automatically syncs with package version
- **GitHub releases**: Automatically triggered by git tags

## Release Process

### 1. Prepare for Release

1. **Ensure all changes are committed** and the `master` branch is up to date:
   ```bash
   git checkout master
   git pull origin master
   ```

2. **Run tests** to ensure everything is working:
   ```bash
   pytest tests/
   ```

3. **Update documentation** if needed and test notebooks:
   ```bash
   cd docs
   sphinx-build -b html . _build/html
   ```

### 2. Version Synchronization

1. **Sync version across all files** (optional, for consistency):
   ```bash
   python scripts/sync_version.py 0.5.2  # Replace with your target version
   ```

2. **Review and commit any changes**:
   ```bash
   git add -A
   git commit -m "Bump version to 0.5.2"
   ```

### 3. Create Release

1. **Create and push git tag**:
   ```bash
   git tag v0.5.2
   git push origin v0.5.2
   ```

2. **The automated systems will handle the rest**:
   - GitHub Actions will build and publish to PyPI
   - Documentation will be updated on ReadTheDocs
   - GitHub release will be created with changelog

### 4. Manual GitHub Release (Alternative)

If you prefer to create releases manually through GitHub UI:

1. Go to [GitHub Releases](https://github.com/routhleck/canns/releases)
2. Click "Create a new release"
3. Choose the tag (or create new one): `v0.5.2`
4. Fill in release title: `Release 0.5.2`
5. The automation will still trigger PyPI publication

## Automated Workflows

### On Tag Push (`v*`)
- **Build package** and verify version consistency
- **Publish to PyPI** (requires `PYPI_API_TOKEN` secret)
- **Update documentation** on ReadTheDocs
- **Create GitHub release** with auto-generated changelog

### On Documentation Changes
- **Build docs** to check for errors
- **Test notebooks** for syntax issues
- **Upload artifacts** for review

## Version Detection Logic

The system uses multiple fallbacks to ensure version consistency:

1. **Git tags** (primary source): `git describe --tags --abbrev=0`
2. **Package metadata**: `importlib.metadata.version("canns")`
3. **Fallback**: Hardcoded version in source files

## Configuration Files

- **`pyproject.toml`**: Dynamic versioning configuration
- **`docs/conf.py`**: Documentation version detection
- **`.github/workflows/release.yml`**: Release automation
- **`scripts/sync_version.py`**: Version synchronization tool

## Required Secrets

For full automation, configure these GitHub repository secrets:

- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `READTHEDOCS_TOKEN`: ReadTheDocs API token (optional)

## Troubleshooting

### Version Mismatch
If you encounter version mismatches, run:
```bash
python scripts/sync_version.py $(git describe --tags --abbrev=0 | sed 's/v//')
```

### Failed PyPI Upload
- Check that `PYPI_API_TOKEN` is correctly configured
- Ensure version doesn't already exist on PyPI
- Verify package builds correctly: `python -m build`

### Documentation Build Errors
- Test locally: `cd docs && sphinx-build -b html . _build/html`
- Check ReadTheDocs build logs
- Verify all dependencies are in `docs/requirements.txt`

## Version History

- `v0.5.1`: Current stable release
- `v0.5.0`: Major feature additions
- `v0.4.1`: Bug fixes and improvements
- `v0.4.0`: Initial public release

## Best Practices

1. **Semantic Versioning**: Follow [SemVer](https://semver.org/) (MAJOR.MINOR.PATCH)
2. **Clear Changelog**: Document what changed in each release
3. **Test Before Release**: Always run full test suite
4. **Documentation Updates**: Keep docs in sync with code changes
5. **Breaking Changes**: Bump major version for API changes

---

*This release process ensures consistent, automated, and reliable software delivery.*