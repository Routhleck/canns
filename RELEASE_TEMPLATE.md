# Release Template

Use this template to create release notes for each version.

---

## Release vX.Y.Z

**Release Date:** YYYY-MM-DD

### Summary

Brief 1-2 sentence overview of what this release accomplishes.

### Major Features

New capabilities or significant enhancements:

- **Feature Name**: Description of the feature and its benefits
- **Feature Name**: Description of the feature and its benefits

### Improvements

Enhancements to existing functionality:

- **Component/Module**: What was improved and why it matters
- **Performance**: Any performance optimizations
- **API**: API improvements or refinements

### Bug Fixes

Issues resolved in this release:

- Fixed: Description of the bug and the fix
- Fixed: Description of the bug and the fix

### Breaking Changes

⚠️ **Important**: Changes that may require code updates:

- **Change Description**: What changed and how to migrate
- **Change Description**: What changed and how to migrate

### Deprecations

Features marked for future removal:

- **Deprecated Feature**: Use `new_feature()` instead. Will be removed in vX.Y.Z
- **Deprecated Feature**: Use `new_feature()` instead. Will be removed in vX.Y.Z

### Documentation

Documentation updates and additions:

- Added: New guides or tutorials
- Updated: Documentation improvements
- Fixed: Documentation corrections

### Dependencies

Changes to project dependencies:

- Added: `package-name>=X.Y.Z` - Reason for addition
- Updated: `package-name` from X.Y.Z to X.Y.Z - Reason for update
- Removed: `package-name` - Reason for removal

### Internal Changes

Changes that don't affect external API:

- Refactoring, testing, or infrastructure improvements
- Build system or CI/CD updates

---

## Example Release Notes

### Release v0.6.0

**Release Date:** 2025-10-22

### Summary

This release adds place cell theta sweep visualization and refactors navigation tasks with geodesic distance computation for complex environments.

### Major Features

- **Place Cell Network & Theta Sweep**: Implemented `PlaceCellNetwork` using graph-based continuous-attractor dynamics with `create_theta_sweep_place_cell_animation()` for visualizing place cell activity during navigation
- **Geodesic Distance Computation**: Added geodesic distance support for complex environments (T-maze, obstacles) via `MovementCostGrid` and `GeodesicDistanceResult` classes
- **T-Maze Navigation Tasks**: New T-maze variants with optional recesses at junctions for both open-loop and closed-loop navigation

### Improvements

- **Navigation Architecture**: Extracted common functionality into `BaseNavigationTask` base class for better code reuse
- **Visualization**: Enhanced environment visualization with movement cost overlays and unified animation interface
- **Grid Indexing**: Improved coordinate-to-grid mapping across models and tasks

### Bug Fixes

- Fixed: Grid resolution validation now properly checks for positive values
- Fixed: Trajectory analysis now correctly handles edge cases with zero angular velocity

### Breaking Changes

None - all changes are backward compatible.

### Documentation

- Added: Example `theta_sweep_place_cell_network.py` demonstrating place cell animations
- Updated: Navigation task documentation with geodesic distance examples
- Added: API reference for `PlaceCellNetwork` and geodesic utilities

### Dependencies

No dependency changes in this release.

### Internal Changes

- Refactored: Navigation tasks now share common base class
- Testing: Added tests for geodesic distance computation
- CI/CD: Updated workflows for multiprocessing-safe examples

---

## Release Checklist

Before creating a release, ensure:

- [ ] All tests pass (`pytest tests/`)
- [ ] Documentation builds without errors (`make docs`)
- [ ] Examples run successfully
- [ ] Version number follows semantic versioning
- [ ] CHANGELOG.md is updated
- [ ] Breaking changes are clearly documented
- [ ] Migration guide provided (if applicable)
- [ ] All dependencies are up to date

## Quick Release Commands

```bash
# 1. Sync version (optional)
python scripts/sync_version.py X.Y.Z

# 2. Commit version bump
git add -A
git commit -m "Release vX.Y.Z"

# 3. Create and push tag
git tag vX.Y.Z
git push origin vX.Y.Z

# Automation handles PyPI publish and docs update
```

---

*Follow semantic versioning: MAJOR.MINOR.PATCH*
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible