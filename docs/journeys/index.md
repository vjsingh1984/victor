# User Journey Paths

Choose your journey based on your role and goals. Each journey provides a structured learning path with clear outcomes.

```mermaid
flowchart LR
    Start([Choose Your Path]) --> Beginner[Beginner<br/>New Users]
    Start --> Intermediate[Intermediate<br/>Daily Users]
    Start --> Developer[Developer<br/>Contributors]
    Start --> Operations[Operations<br/>DevOps/SRE]
    Start --> Advanced[Advanced<br/>Architects]

    Beginner --> BOut[30 min<br/>Basic Usage]
    Intermediate --> IOut[80 min<br/>Power User]
    Developer --> DOut[2 hours<br/>Contributor]
    Operations --> OOut[80 min<br/>Production]
    Advanced --> AOut[2.5 hours<br/>Expert]

    style Beginner fill:#e8f5e9,stroke:#2e7d32
    style Intermediate fill:#e3f2fd,stroke:#1565c0
    style Developer fill:#fff3e0,stroke:#e65100
    style Operations fill:#f3e5f5,stroke:#6a1b9a
    style Advanced fill:#fce4ec,stroke:#880e4f
```text

## 🚀 Beginner Journey

**For:** New users installing Victor for the first time
**Time:** 30 minutes
**Outcome:** Ready for daily use

**You'll Learn:**
- Install Victor (pipx, Docker, development)
- Configure providers (cloud and local)
- Have your first conversation
- Use basic features
- Switch providers mid-conversation

→ [Start Beginner Journey](beginner.md)

---

## 💡 Intermediate Journey

**For:** Daily users who want advanced features
**Time:** 80 minutes
**Prerequisite:** [Beginner Journey](beginner.md)
**Outcome:** Power user

**You'll Learn:**
- Master 55+ tools
- Create YAML workflows
- Use multi-agent teams
- Customize profiles
- Leverage advanced modes (PLAN, EXPLORE)

→ [Start Intermediate Journey](intermediate.md)

---

## 👨‍💻 Developer Journey

**For:** Contributors building extensions
**Time:** 2 hours
**Prerequisite:** [Intermediate Journey](intermediate.md)
**Outcome:** Active contributor

**You'll Learn:**
- Set up development environment
- Write and run tests
- Create custom tools
- Develop verticals
- Contribute to core

→ [Start Developer Journey](developer.md)

---

## 🔧 Operations Journey

**For:** DevOps/SRE deploying to production
**Time:** 80 minutes
**Prerequisite:** Docker/Kubernetes experience
**Outcome:** Production deployment

**You'll Learn:**
- Deploy with Docker and Kubernetes
- Configure monitoring and alerting
- Implement security controls
- Meet compliance requirements (SOC2, GDPR)
- Scale Victor for enterprise

→ [Start Operations Journey](operations.md)

---

## 🏗️ Advanced Journey

**For:** Architects evaluating system design
**Time:** 2.5 hours
**Prerequisite:** [Developer Journey](developer.md)
**Outcome:** System expert

**You'll Learn:**
- Two-layer coordinator architecture
- Protocol-based design principles
- Event-driven communication
- Dependency injection patterns
- Performance optimization strategies

→ [Start Advanced Journey](advanced.md)

---

## Not Sure Where to Start?

### Quick Decision Guide

| Your Goal | Start With |
|-----------|------------|
| "I want to try Victor" | → [Beginner Journey](beginner.md) |
| "I use Victor daily, want more" | → [Intermediate Journey](intermediate.md) |
| "I want to contribute code" | → [Developer Journey](developer.md) |
| "I need to deploy Victor" | → [Operations Journey](operations.md) |
| "I want to understand the architecture" | → [Advanced Journey](advanced.md) |
| "I'm evaluating Victor for my team" | → [Beginner Journey](beginner.md), then [Advanced Journey](advanced.md) |

### By Role

| Role | Recommended Journey |
|------|-------------------|
| **Individual Developer** | Beginner → Intermediate |
| **Software Engineer** | Beginner → Intermediate → Developer |
| **Tech Lead / Architect** | Beginner → Advanced |
| **DevOps Engineer** | Beginner → Operations |
| **Full Stack Developer** | Beginner → Intermediate → Developer → Operations |
| **Engineering Manager** | Beginner → Advanced (evaluation) |

### By Time Available

| Time Available | Journey |
|----------------|---------|
| **30 minutes** | [Beginner](beginner.md) - Get started quickly |
| **1 hour** | [Beginner](beginner.md) + skim [Intermediate](intermediate.md) |
| **2 hours** | [Beginner](beginner.md) + [Intermediate](intermediate.md) |
| **4 hours** | Beginner + Intermediate + Developer |
| **Half day** | All journeys (comprehensive understanding) |

---

## Journey Navigation

Each journey includes:
- ✅ Clear learning objectives
- ⏱️ Time estimates for each section
- 📊 Visual diagrams (Mermaid.js)
- 🎖️ Practical exercises
- ➡️ "What's Next" recommendations
- 🔗 Links to detailed documentation

### Journey Features

**Visual Diagrams:**
- [Beginner Onboarding](../diagrams/user-journeys/beginner-onboarding.mmd)
- [Contributor Workflow](../diagrams/user-journeys/contributor-workflow.mmd)
- [Coordinator Architecture](../diagrams/architecture/coordinator-layers.mmd)
- [Deployment Patterns](../diagrams/operations/deployment.mmd)

**Progress Tracking:**
Each journey has clear milestones:
- Step completion checkboxes
- Knowledge verification points
- Practical skill assessments

---

## Additional Resources

### Quick Reference Cards

- [CLI Commands](../user-guide/cli-reference.md)
- [Configuration Options](../reference/configuration/)
- [Tools Reference](../user-guide/tools.md)

### Troubleshooting

- [FAQ](../user-guide/faq.md)
- [Common Issues](../getting-started/troubleshooting.md)
- [Debug Guide](../contributing/debugging.md)

### Community

- [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)
- [Contributing Guide](../contributing/)

---

**Last Updated:** January 31, 2026
**Total Reading Time:** 5 hours (all journeys)

Need help? Start with the [Beginner Journey](beginner.md) or open a [GitHub
  Discussion](https://github.com/vjsingh1984/victor/discussions).
