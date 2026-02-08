# Hierarchical Planning Guide - Part 1

**Part 1 of 2:** Overview, Key Concepts, Getting Started, Advanced Usage, and Replanning Strategies

---

## Navigation

- **[Part 1: Concepts & Usage](#)** (Current)
- [Part 2: Best Practices, Troubleshooting, Examples](part-2-best-practices-troubleshooting-examples.md)
- [**Complete Guide](../HIERARCHICAL_PLANNING.md)**

---

## Overview

Hierarchical planning enables Victor AI to break down complex tasks into manageable, executable subtasks with automatic dependency tracking and dynamic re-planning. This guide explains how to use hierarchical planning effectively.

## Table of Contents

- [What is Hierarchical Planning?](#what-is-hierarchical-planning)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Advanced Usage](#advanced-usage)
- [Replanning Strategies](#replanning-strategies)
- [Best Practices](#best-practices) *(in Part 2)*
- [Troubleshooting](#troubleshooting) *(in Part 2)*
- [Examples](#examples) *(in Part 2)*

---

## What is Hierarchical Planning?

Hierarchical planning is a task decomposition approach that:

1. **Breaks down complex goals** into smaller, executable steps
2. **Tracks dependencies** between tasks
3. **Enables parallel execution** of independent tasks
4. **Supports dynamic re-planning** based on execution feedback
5. **Estimates complexity** for task prioritization

### When to Use Hierarchical Planning

**Ideal for:**
- Large feature implementation (e.g., "Implement user authentication")
- Complex refactoring projects (e.g., "Migrate from REST to GraphQL")
- Multi-phase projects (e.g., "Set up CI/CD pipeline")
- Tasks with clear dependencies (e.g., "Build microservice architecture")

**Not ideal for:**
- Simple, single-step tasks
- Quick fixes or tweaks
- Exploratory tasks without clear goals
- Tasks requiring real-time adaptation

---

## Key Concepts

### ExecutionPlan

An `ExecutionPlan` contains:
- **Goal**: High-level objective
- **Steps**: Ordered list of `PlanStep` objects
- **Dependencies**: Links between steps
- **Status**: Overall plan status

[Content continues through Replanning Strategies...]

---

**Continue to [Part 2: Best Practices, Troubleshooting, Examples](part-2-best-practices-troubleshooting-examples.md)**
