# Victor AI Release Checklist

This checklist ensures a smooth, controlled release process for Victor AI.

## Pre-Release Checklist

### Code Quality
- [ ] All tests passing (unit + integration)
  - [ ] Unit tests: `pytest tests/unit -v`
  - [ ] Integration tests: `pytest tests/integration -v`
  - [ ] End-to-end tests: `pytest tests/integration/test_end_to_end_workflows.py -v`
  - [ ] Test coverage > 80%
- [ ] Code formatting complete
  - [ ] `black victor tests --check`
  - [ ] `ruff check victor tests`
  - [ ] `mypy victor` (gradual adoption)
- [ ] Security scan clean
  - [ ] `bandit -r victor`
  - [ ] `safety check`
  - [ ] `pip-audit`
  - [ ] `semgrep victor`

### Documentation
- [ ] CHANGELOG.md updated with version and release date
- [ ] RELEASE_NOTES.md complete with all sections
- [ ] VERSION file updated with version, build SHA, and date
- [ ] README.md version references updated
- [ ] API documentation generated: `victor-generate-docs`
- [ ] Architecture documentation up to date
- [ ] Migration guide updated (if breaking changes)

### Configuration
- [ ] pyproject.toml version updated
- [ ] Dependencies verified and up to date
- [ ] No deprecated features used
- [ ] Environment variables documented
- [ ] YAML configuration files validated

### Testing
- [ ] Smoke tests pass on all platforms
  - [ ] macOS (ARM64)
  - [ ] macOS (x64)
  - [ ] Linux (x64)
  - [ ] Windows (x64)
- [ ] Performance benchmarks run and documented
  - [ ] Startup time < 500ms (with lazy loading)
  - [ ] Tool selection latency < 20ms
  - [ ] Memory usage acceptable
- [ ] Security tests pass (> 95% pass rate)
- [ ] Load tests pass (if applicable)

### Build Verification
- [ ] Clean build from scratch
  ```bash
  rm -rf dist/ build/ *.egg-info
  python -m build
  ```
- [ ] Source distribution (sdist) builds correctly
- [ ] Wheel builds correctly
- [ ] Installation works: `pip install dist/victor_ai-*.whl`
- [ ] Import works: `python -c "import victor; print(victor.__version__)"`

### Git Preparation
- [ ] All changes committed
- [ ] Working directory clean: `git status`
- [ ] On main branch (or release branch)
- [ ] Release branch merged to main (if using release branches)
- [ ] No merge conflicts

## Release Execution

### Automated Release
Run the release script:
```bash
./scripts/release.sh 1.0.0
```

Or manual steps below:

### Version Bump
- [ ] Update version in pyproject.toml
- [ ] Update VERSION file
- [ ] Commit version bump: `git commit -m "chore: bump version to 1.0.0"`

### Build Packages
- [ ] Build source distribution: `python -m build --sdist`
- [ ] Build wheel: `python -m build --wheel`
- [ ] Verify packages in `dist/`

### Generate Checksums
- [ ] Run: `python scripts/create_checksums.py`
- [ ] Verify SHA256SUMS file created
- [ ] (Optional) Sign with GPG: `gpg --detach-sign --armor SHA256SUMS`

### Git Tag
- [ ] Create annotated tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] Verify tag on GitHub

### PyPI Upload
- [ ] Test upload to TestPyPI first (optional):
  ```bash
  twine upload --repository testpypi dist/*
  ```
- [ ] Upload to PyPI:
  ```bash
  twine upload dist/*
  ```
- [ ] Verify on PyPI: https://pypi.org/project/victor-ai/

### GitHub Release
- [ ] Go to GitHub Releases page
- [ ] Click "Draft a new release"
- [ ] Choose tag: v1.0.0
- [ ] Release title: "Victor AI 1.0.0"
- [ ] Copy RELEASE_NOTES.md content to release description
- [ ] Attach distribution files (optional)
- [ ] Attach SHA256SUMS file
- [ ] Click "Publish release"

### Package Managers (Optional)
- [ ] Homebrew formula updated (if applicable)
- [ ] Debian/Ubuntu packages built (if applicable)
- [ ] RPM packages built (if applicable)
- [ ] Docker images pushed (if applicable)

## Post-Release Verification

### Installation Verification
- [ ] Clean install from PyPI works:
  ```bash
  pip install victor-ai==1.0.0
  ```
- [ ] Version check works: `victor --version`
- [ ] Health check passes: `victor --health-check`
- [ ] Basic functionality works: `victor chat --no-tui --query "test"`

### Integration Verification
- [ ] VS Code extension works with new version
- [ ] JetBrains extension works with new version
- [ ] API server starts correctly: `victor serve`
- [ ] MCP server works: `victor mcp`

### Documentation Verification
- [ ] GitHub release notes visible
- [ ] PyPI page shows correct version
- [ ] Documentation links work
- [ ] Examples run successfully

### Monitoring
- [ ] Monitor GitHub Issues for release-related issues
- [ ] Monitor PyPI download stats
- [ ] Monitor error reports (if error tracking enabled)
- [ ] Monitor performance metrics

## Rollback Procedure

If critical issues are found:

### PyPI Rollback
- [ ] **Note**: PyPI does not allow deleting releases, only yanking
- [ ] Yank the release if critical: `twine yank victor-ai==1.0.0`
- [ ] Upload hotfix as 1.0.1

### Git Tag Rollback
- [ ] Delete local tag: `git tag -d v1.0.0`
- [ ] Delete remote tag: `git push origin :refs/tags/v1.0.0`
- [ ] Create new tag for fixed version

### GitHub Release Rollback
- [ ] Edit release to mark as "Pre-release"
- [ ] Add warning about critical issues
- [ ] Delete and recreate with hotfix version

### Communication
- [ ] Post announcement about issues
- [ ] Provide workaround or fix timeline
- [ ] Update documentation with known issues

## Post-Release Tasks

### Announcements
- [ ] Post announcement on GitHub Discussions
- [ ] Update project website (if applicable)
- [ ] Post on social media (Twitter, LinkedIn, etc.)
- [ ] Send email newsletter (if applicable)

### Metrics
- [ ] Track PyPI downloads
- [ ] Track GitHub stars/forks
- [ ] Track GitHub issues/PRs
- [ ] Collect user feedback

### Next Version Planning
- [ ] Create milestone for next version
- [ ] Roadmap updated
- [ ] Backlog prioritized
- [ ] Release target date set

### Maintenance
- [ ] Monitor for bug reports
- [ ] Respond to issues promptly
- [ ] Backport critical fixes to release branch (if applicable)
- [ ] Prepare for patch release if needed

## Quick Reference

### Essential Commands
```bash
# Run all tests
make test-all

# Run security scans
make lint
bandit -r victor
safety check
pip-audit

# Build packages
python -m build

# Generate checksums
python scripts/create_checksums.py

# Create git tag
git tag -a v1.0.0 -m "Release 1.0.0"

# Upload to PyPI
twine upload dist/*

# Automated release
./scripts/release.sh 1.0.0
```

### File Locations
- CHANGELOG: `CHANGELOG.md`
- Release notes: `RELEASE_NOTES.md`
- Version: `VERSION`
- Configuration: `pyproject.toml`
- Debian: `debian/`
- RPM spec: `victor-ai.spec`
- Release script: `scripts/release.sh`
- Checksums: `scripts/create_checksums.py`

### Critical URLs
- PyPI: https://pypi.org/project/victor-ai/
- GitHub: https://github.com/vijayksingh/victor
- Releases: https://github.com/vijayksingh/victor/releases
- Issues: https://github.com/vijayksingh/victor/issues

---

## Notes

- Always test in a clean virtual environment
- Never skip tests unless absolutely necessary
- Always have a rollback plan
- Communicate early and often
- Monitor after release closely
- Learn from each release

**Remember**: A good release is boring. No news is good news!
