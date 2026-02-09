# Victor AI - Troubleshooting Guide - Part 1

**Part 1 of 2:** Installation Issues, Provider Connection, Performance, Tools, Memory, Configuration, and Debugging Tips

---

## Navigation

- **[Part 1: Common Issues](#)** (Current)
- [Part 2: Error Messages, Getting Help, Template](part-2-errors-help-template.md)
- [**Complete Guide](../TROUBLESHOOTING.md)**

---

This guide helps you diagnose and fix common issues with Victor AI.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Provider Connection Issues](#provider-connection-issues)
3. [Performance Issues](#performance-issues)
4. [Tool Issues](#tool-issues)
5. [Memory Issues](#memory-issues)
6. [Configuration Issues](#configuration-issues)
7. [Debugging Tips](#debugging-tips)
8. [Common Error Messages](#common-error-messages) *(in Part 2)*
9. [Getting Additional Help](#getting-additional-help) *(in Part 2)*
10. [Issue Template](#issue-template) *(in Part 2)*

---

## Installation Issues

### Issue: "No module named 'victor'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'victor'
```

**Solutions:**

1. **Install in development mode:**
```bash
cd /path/to/victor
pip install -e ".[dev]"
```

2. **Check Python path:**
```bash
which python
python -c "import sys; print(sys.path)"
```

3. **Reinstall Victor:**
```bash
pip uninstall victor-ai
pip install -e ".[dev]"
```

[Content continues through Debugging Tips...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Error Messages, Getting Help, Template](part-2-errors-help-template.md)**
