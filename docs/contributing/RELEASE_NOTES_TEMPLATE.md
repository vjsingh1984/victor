# Release Notes Template

Copy this template for each new release. Update the `[version]` placeholder and fill in the sections.

---

## [Version] - YYYY-MM-DD

### Summary

Brief description of this release (2-3 sentences).

### Highlights

- **[Feature 1]**: Brief description
- **[Feature 2]**: Brief description
- **[Improvement 1]**: Brief description
- **[Bug Fix 1]**: Brief description

### What's New

#### Features
- **[Category]: [Feature name]**
  - Description of the new feature
  - Benefits/value to users
  - Example usage or documentation link

#### Enhancements
- **[Category]: [Enhancement name]**
  - Description of the improvement
  - Impact on existing functionality

#### Performance
- **[Improvement]**: Description
  - Before: X seconds/memory
  - After: Y seconds/memory
  - Improvement: Z%

### Bug Fixes

- **[Bug ID or Title]**: Description of the fix
  - Issue: #[number]
  - Impact: Who this affects and how

### Breaking Changes

⚠️ **IMPORTANT**: If there are breaking changes, describe them clearly:

- **[Change Name]**: Description
  - Why: Reason for the change
  - Migration: Steps to update code
  - Example:
    ```python
    # Old code
    old_method()

    # New code
    new_method()
    ```

### Deprecations

- **[Feature/API]**: Description
  - Deprecated in: [Version]
  - To be removed in: [Version]
  - Migration path: Link to documentation

### Documentation

- Added: [New documentation]
- Updated: [Updated documentation]
- Improved: [Documentation improvements]

### Contributors

Thank you to the following contributors:
- @[username1] - [Contribution]
- @[username2] - [Contribution]

### Installation

```bash
# From PyPI
pip install --upgrade victor-ai

# From source
pip install git+https://github.com/vjsingh1984/victor.git

# With optional dependencies
pip install --upgrade "victor-ai[all]"
```

### Upgrade Instructions

If you're upgrading from [previous version]:

1. Review breaking changes above
2. Update your code if needed
3. Run: `pip install --upgrade victor-ai`
4. Test your workflows
5. Report issues at: https://github.com/vjsingh1984/victor/issues

### Configuration Changes

If there are new configuration options:

```yaml
# config.yaml or profiles.yaml
new_option:
  setting: value
  description: "Description"
```

### Known Issues

- **[Issue 1]**: Description
  - Workaround: How to work around it
  - Status: Being tracked in #[number]

### Testing

- Unit tests: [X/Y] passing
- Integration tests: [X/Y] passing
- Coverage: [X]%

### SHA256 Checksums

For verification of release artifacts:

```
victor-ai-[version]-py3-none-any.whl: SHA256
victor-ai-[version].tar.gz: SHA256
```

### Full Changelog

For complete list of changes, see:
- [GitHub Commits](https://github.com/vjsingh1984/victor/commits/v[version])
- [MIGRATION.md](docs/MIGRATION.md) for breaking changes
- [CHANGELOG.md](CHANGELOG.md) for full history

---

## Previous Releases

See [CHANGELOG.md](CHANGELOG.md) for details on previous releases.

---

## Template Usage Guide

### Summary Section
- Keep it concise (2-3 sentences max)
- Focus on user-facing value

### Highlights Section
- List 3-5 top items
- Use emojis for visual appeal:
  - ✨ for new features
  - 🚀 for performance
  - 🐛 for bug fixes
  - 💥 for breaking changes
  - 📚 for documentation

### What's New Section
- Group by category
- Use bullet points with bold prefixes
- Link to documentation when available

### Breaking Changes Section
- Use ⚠️ emoji to draw attention
- Provide migration code examples
- Link to migration guide

### Version Numbering
- Follow semantic versioning: MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Contributors Section
- Use GitHub usernames
- List all contributors (including non-code)
- Link to their PRs if desired

### Example Release Note

```markdown
## 0.5.0 - 2025-01-10

### Summary
This release introduces comprehensive documentation restructuring,
benchmark vertical fixes, and GitHub Pages deployment automation.

### Highlights
- **📚 Documentation**: Complete restructuring with 105-page site
- **🔧 Benchmark**: Fixed missing capabilities and prompts
- **🚀 Deployment**: Automated GitHub Pages via GitHub Actions
- **📖 Migration**: Added upgrade guide for breaking changes
```
