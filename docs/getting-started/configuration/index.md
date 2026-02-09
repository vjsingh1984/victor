# Configuration Guide

Guide to configuring Victor AI for your development environment.

---

## Quick Summary

This guide covers all aspects of Victor AI configuration:

- **Configuration Overview** - Configuration system architecture
- **Quick Setup** - Basic configuration in 5 minutes
- **Configuration Locations** - Where configuration files live
- **Environment Variables** - Environment-based configuration
- **Profiles** - Multiple configuration profiles
- **Global Settings** - Application-wide settings
- **Project Context** - Project-specific configuration (.victor.md)
- **Modes** - Agent mode configuration (build, plan, explore)
- **MCP Integration** - Model Context Protocol setup
- **Examples** - Common configuration scenarios
- **Validation** - Testing and validating configuration
- **Troubleshooting** - Common configuration issues

---

## Guide Parts

### [Part 1: Configuration Basics](part-1-configuration-basics.md)
- Configuration Overview
- Quick Setup
- Configuration Locations
- Environment Variables
- Profiles Configuration
- Global Settings

### [Part 2: Project Configuration](part-2-project-configuration.md)
- Project Context Files
  - Overview, Tech Stack, Code Style
  - Testing, Directories, Conventions
  - Victor Instructions
- Modes Configuration
- MCP Configuration
- Example Configurations

### [Part 3: Validation & Best Practices](part-3-validation-best-practices.md)
- Validation and Testing
- Troubleshooting
- Best Practices
- Next Steps

---

## Quick Start

**1. Create configuration file:**
```bash
mkdir -p ~/.victor
cat > ~/.victor/config.yaml << EOF
provider: anthropic
model: claude-sonnet-4-5
mode: build
EOF
```text

**2. Set API key:**
```bash
export ANTHROPIC_API_KEY=sk-your-key-here
```

**3. Test configuration:**
```bash
victor chat "Hello, Victor!"
```text

---

## Configuration Priority

Victor loads configuration in this order (later overrides earlier):

1. **Defaults** - Built-in defaults
2. **Global config** - `~/.victor/config.yaml`
3. **Profile config** - `~/.victor/profiles/<name>.yaml`
4. **Project config** - `.victor.md` in project root
5. **Environment variables** - `VICTOR_*` env vars
6. **Command-line args** - Highest priority

---

## Related Documentation

- [Configuration Reference](../../reference/configuration/index.md)
- [Environment Variables](../../reference/api/CONFIGURATION_REFERENCE.md)
- [Project Context](../project-context/README.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 15 min (all parts)
