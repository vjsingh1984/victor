# Victor AI - Developer Guide Video Script

**Title:** Building with Victor AI: Creating Custom Verticals and Extensions
**Duration:** 12-15 minutes
**Target Audience:** Plugin developers, vertical creators, contributors
**Prerequisites:** Watch Advanced Features video

---

## Video Outline

1. **Introduction** (0:00-1:00)
2. **Vertical Architecture** (1:00-3:00)
3. **Creating a Vertical** (3:00-6:00)
4. **Vertical Configuration** (6:00-8:00)
5. **Testing & Publishing** (8:00-10:00)
6. **Contributing Guidelines** (10:00-11:00)
7. **Community & Support** (11:00-12:00)
8. **Conclusion** (12:00-12:30)

---

## Script

### [0:00-1:00] Introduction

**[Visual: Developer-focused montage]**

**Narrator:** "Welcome to the Victor AI Developer Guide. This video is for developers who want to extend Victor with custom verticals, contribute to the project, or build Victor-powered applications."

**[Visual: Extension ecosystem]**

**Narrator:** "Victor's plugin architecture lets you create domain-specific verticals, custom tools, and workflows. Today, we'll build a complete vertical from scratch."

### [1:00-3:00] Vertical Architecture

#### [1:00-2:00] Understanding Verticals

**[Visual: Vertical architecture diagram]**

**Narrator:** "Verticals are domain-specific specializations. Each vertical provides custom tools, system prompts, workflows, and configuration."

**[Visual: Built-in verticals]**

**Narrator:** "Victor includes five built-in verticals: coding, DevOps, RAG, data analysis, and research. Each is optimized for its domain."

**[Visual: Vertical components diagram]**

**Narrator:** "A vertical consists of: a VerticalBase class, tools directory, workflows, configuration files, and tests."

#### [2:00-3:00] The Vertical Base Class

**[Visual: VerticalBase code]**

**Narrator:** "All verticals inherit from VerticalBase. Implement get_tools() to return your tools, and get_system_prompt() for domain-specific prompts."

**[Visual: Vertical lifecycle]**

**Narrator:** "Victor automatically discovers verticals via entry points, initializes them, and makes their tools available to agents."

### [3:00-6:00] Creating a Vertical

#### [3:00-4:00] Project Setup

**[Visual: Directory structure creation]**

**Narrator:** "Let's create a security analysis vertical. Start with the directory structure:"

```
victor-security/
├── victor_security/
│   ├── __init__.py
│   ├── vertical.py
│   ├── tools/
│   ├── workflows/
│   └── config/
├── tests/
└── pyproject.toml
```

**[Visual: pyproject.toml]**

**Narrator:** "Configure pyproject.toml with the entry point for your vertical:"

```toml
[project.entry-points."victor.verticals"]
security = "victor_security.vertical:SecurityVertical"
```

#### [4:00-5:00] Implementing the Vertical Class

**[Visual: SecurityVertical code]**

**Narrator:** "Create your vertical class. Here's our SecurityVertical:"

**[Visual: Code walkthrough]**

**Narrator:** "The get_tools() method imports and returns security tools. The get_system_prompt() provides security-focused guidance to the AI."

**[Visual: Tool example]**

**Narrator:** "Each tool is a class inheriting from BaseTool. This VulnerabilityScanner tool detects security issues."

#### [5:00-6:00] Adding Tools and Workflows

**[Visual: Multiple tools]**

**Narrator:** "Add complementary tools: DependencyAnalyzer for known vulnerabilities, ComplianceChecker for standards compliance."

**[Visual: Workflow YAML]**

**Narrator:** "Create workflows that combine tools. This security audit workflow runs multiple scans and generates a comprehensive report."

### [6:00-8:00] Vertical Configuration

#### [6:00-7:00] Modes Configuration

**[Visual: modes.yaml]**

**Narrator:** "Configure vertical-specific modes in YAML. Each mode has exploration level, edit permissions, and tool budget multipliers."

**[Visual: Mode switching demo]**

**Narrator:** "Users switch modes to control the AI's behavior. Audit mode for thorough reviews, scan mode for quick checks."

#### [7:00-8:00] Capabilities and Teams

**[Visual: capabilities.yaml]**

**Narrator:** "Define capabilities in YAML. Each capability has a type, description, and handler."

**[Visual: teams.yaml]**

**Narrator:** "Configure teams for multi-agent workflows. Specify roles, personas, and capabilities."

### [8:00-10:00] Testing & Publishing

#### [8:00-9:00] Testing Your Vertical

**[Visual: Test code]**

**Narrator:** "Comprehensive testing is essential. Test the vertical class, individual tools, and integration with agents."

**[Visual: Test execution]**

**Narrator:** "Run tests with pytest. Victor provides fixtures for mocking providers and tools."

**[Visual: Coverage report]**

**Narrator:** "Aim for high test coverage. Victor's CI checks coverage on all pull requests."

#### [9:00-10:00] Publishing

**[Visual: Build and publish commands]**

**Narrator:** "Publish your vertical to PyPI. Build the distribution, upload to PyPI, and users install it with pip."

**[Visual: Installation and usage]**

**Narrator:** "Users install your vertical and it's automatically available in Victor."

### [10:00-11:00] Contributing Guidelines

**[Visual: Contribution workflow]**

**Narrator:** "Contributing to Victor core? Follow these guidelines:"

**[Visual: Code standards]**

**Narrator:** "Follow PEP 8, use type hints, write docstrings. Run black, ruff, and mypy before committing."

**[Visual: Testing requirements]**

**Narrator:** "All contributions need tests. Aim for 80%+ coverage. Include both unit and integration tests."

**[Visual: Documentation]**

**Narrator:** "Update documentation for new features. Include examples and use cases."

### [11:00-12:00] Community & Support

**[Visual: Community resources]**

**Narrator:** "Join the Victor community:

**[Visual: GitHub]**
- GitHub for issues and pull requests

**[Visual: Discord]**
- Discord for real-time chat

**[Visual: Documentation]**
- Comprehensive docs at docs.victor.ai

**[Visual: Examples]**
- Example verticals in the repository"

### [12:00-12:30] Conclusion

**[Visual: Recap]**

**Narrator:** "Victor's extensible architecture makes it easy to create domain-specific verticals. Whether you're building a security analyzer, a DevOps toolkit, or a research assistant, Victor provides the foundation."

**[Visual: Call to action]**

**Narrator:** "Start building today. Check out the example verticals, read the development documentation, and join our community of contributors. We can't wait to see what you'll create."

**[Visual: Links]**

**Narrator:** "Victor AI - Built by developers, for developers. Let's build the future of AI-assisted development together."

---

## Production Notes

### Technical Depth
- Most technical of the three videos
- Assumes programming knowledge
- Show actual code, not summaries

### Code Visibility
- Large, readable code snippets
- Syntax highlighting
- Line numbers for reference

### Development Environment
- Show VS Code or similar editor
- Terminal for commands
- Browser for documentation

### Pace
- Slower than other videos
- Pause on important concepts
- Allow time to read code

---

## Companion Resources

### Developer Documentation
- Vertical creation: `docs/tutorials/CREATING_VERTICALS.md`
- Tool development: `docs/tutorials/CREATING_TOOLS.md`
- Contribution guide: `CONTRIBUTING.md`

### Example Code
- External vertical example: `examples/external_vertical/`
- Built-in verticals: `victor/coding/`, `victor/devops/`
- Contribution examples: GitHub PRs

### Templates
- Vertical template: `docs/vertical_template_guide.md`
- Tool template: Included in tool tutorial
- Configuration templates: `victor/config/`

---

## Series Summary

This completes the three-video series:

1. **Introduction**: Overview and basic features
2. **Advanced Features**: Workflows, tools, teams
3. **Developer Guide**: Building extensions

Together, these videos provide a complete guide to using and extending Victor AI, from beginner to advanced developer.

---

**End of Script**
