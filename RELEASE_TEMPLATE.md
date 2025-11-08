# Release Template

Use this template for creating GitHub releases. Replace placeholders with actual content.

---

## Title Format
```
Release vX.Y.Z: [Main Theme/Feature Summary]
```

Examples:
- `Release v0.8.2: Plotting API Restoration & Documentation Polish`
- `Release v0.8.0: Theta Sweep Models, Pipeline, and Enhanced Spatial Navigation`

---

## Release Body Template

```markdown
## What's New

**[Emoji] [Bold One-Liner Headline]**
- [Bullet point summary 1]
- [Bullet point summary 2]
- [Bullet point summary 3]

**[Optional Second Headline]**  
- [Additional context or supporting features]

## Major Features / Key Changes

### [Emoji] **[Feature/Change Category 1]** (PR #XX)
- **[Specific component]**: [Description]
- **[Specific component]**: [Description]
- **[Specific component]**: [Description]

```python
# Code example showing usage
from canns.module import Component

# Demonstrative code
component = Component(param=value)
result = component.method()
```

### [Emoji] **[Feature/Change Category 2]** (PR #XX)
- **[Specific component]**: [Description]
- **[Enhanced capability]**: [Description]

```python
# Another usage example
```

### [Emoji] **[Feature/Change Category 3]** (PR #XX)
- **[Component updates]**: [Description]
- **[Improvements]**: [Description]

## [Optional] New Components Added

- `path/to/new/module.py` - [Brief description]
- `path/to/another/module.py` - [Brief description]

## [Optional] Technical Improvements

### [Subsection if needed]
- **[Improvement type]**: [Description]
- **[Optimization]**: [Description]
- **[Enhancement]**: [Description]

## [Optional] New Dependencies
- **[package-name] ([version constraint])**: [Purpose/reason]

## [Optional] Breaking Changes
[None - all additions are backward compatible.] OR [List of breaking changes]

## [Optional] Technical Notes
- [Technical detail 1]
- [Technical detail 2]
- [Compatibility information]

## [Optional] Files Added/Modified
- `path/to/file.py`: [Change description]
- `path/to/another/file.py`: [Change description]

## [Optional] Use Cases
- **[User type]**: [How they benefit]
- **[Another user type]**: [How they benefit]

---

**Full Changelog**: https://github.com/Routhleck/canns/compare/v[PREV]...v[CURRENT]
```

---

## Style Guidelines

### Emojis by Section Type
- 🧠 Brain-inspired models / Neural networks
- 🌊 Wave/oscillation features (theta sweep, etc.)
- 🔧 Fixes / Technical improvements
- 📖 Documentation
- 🎨 Plotting / Visualization
- 🗺️ Spatial navigation / Tasks
- 🚀 Performance / Pipeline
- 📊 Data analysis / Import
- 🎛️ Training / Trainer
- 🎯 Progress / User experience
- ✨ New features
- 🔬 Scientific / Experimental tools
- 📈 Progress / Reporting

### Writing Style
1. **Headlines**: Bold + emoji, action-oriented
2. **Descriptions**: Start with component/feature name in bold, followed by description
3. **Code examples**: Always use syntax highlighting with language tag
4. **PR references**: Include PR numbers in section headers when applicable
5. **Bullet formatting**: Use `-` for lists, `**bold**` for emphasis
6. **Versioning**: Always include "Full Changelog" link comparing previous and current version

### Section Ordering
1. What's New (required)
2. Major Features / Key Changes (required)
3. New Components Added (if applicable)
4. Technical Improvements (if applicable)
5. New Dependencies (if applicable)
6. Breaking Changes (if applicable)
7. Technical Notes (if applicable)
8. Code Examples (if not inline)
9. Files Added/Modified (if helpful)
10. Use Cases (if applicable)
11. Full Changelog link (required)

### Content Priority
- **Minor releases (x.y.Z)**: Focus on fixes, polish, documentation
- **Feature releases (x.Y.0)**: Emphasize new capabilities, breaking changes, major refactors
- Always highlight **user-facing impacts** before internal technical details
- Include **code examples** for new APIs or significant changes
- Mention **compatibility** and **migration guidance** when relevant

---

## Automation Hints for AI

When generating a release from git history:

1. **Analyze commits/PRs between tags**: Extract feature descriptions, breaking changes, new files
2. **Categorize changes**: Group by type (features, fixes, docs, deps, breaking)
3. **Extract code samples**: Look for examples in PR descriptions or new example files
4. **Determine version type**: patch (fixes), minor (features), major (breaking)
5. **Generate headline**: Summarize 2-3 main themes from the changes
6. **Match tone**: Technical but accessible, concise but complete
7. **Include metrics**: File counts, PR numbers, performance improvements when available
8. **Cross-reference**: Link related PRs and issues in descriptions