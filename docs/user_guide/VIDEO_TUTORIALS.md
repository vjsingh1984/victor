# Victor AI Video Tutorials

**Production-ready video scripts for learning Victor AI**

---

## Overview

This document contains 5 video tutorial scripts (5 minutes each) covering different aspects of Victor AI. Each script includes:

- **Screen-by-screen guidance**
- **Script narration**
- **Visual cues**
- **Production notes**
- **Alternative formats**

---

## Video 1: Getting Started with Victor AI (5 minutes)

**Target Audience**: New users
**Goal**: Install Victor and complete first task

### Script Outline

| Time | Scene | Content |
|------|-------|---------|
| 0:00-0:30 | Intro | What is Victor AI |
| 0:30-1:30 | Installation | Install Victor |
| 1:30-2:30 | First Chat | Complete first task |
| 2:30-3:30 | Provider Setup | Configure provider |
| 3:30-4:30 | Common Tasks | Review, tests, docs |
| 4:30-5:00 | Next Steps | Resources and links |

### Detailed Script

#### Scene 1: Introduction (0:00-0:30)

**Visual**:
- Title: "Getting Started with Victor AI"
- Subtitle: "Your AI coding assistant in 5 minutes"
- Show Victor logo

**Narration**:
```
Welcome to Victor AI - your open-source coding assistant that works with
21 different LLM providers. In this 5-minute tutorial, you'll install Victor,
set up your first provider, and complete your first coding task.

Victor supports local models like Ollama for privacy and cloud models like
Claude and GPT-4 for maximum capability. Best of all, it's completely free
to use with your own API keys.

Let's get started!
```

#### Scene 2: Installation (0:30-1:30)

**Visual**:
- Terminal window
- Command: `pipx install victor-ai`
- Show installation progress
- Command: `victor --version`

**Narration**:
```
First, let's install Victor. The recommended way is using pipx, which
installs Victor in an isolated environment.

[Type: pipx install victor-ai]

The installation takes about 30 seconds. Once complete, verify the
installation by checking the version.

[Type: victor --version]

Great! Victor is now installed and ready to use. If you're using pip instead
of pipx, the command is just 'pip install victor-ai'.
```

**Production Notes**:
- Show clear, large text in terminal
- Use light terminal background for better visibility
- Speed up installation progress (cut to completion)

#### Scene 3: First Chat (1:30-2:30)

**Visual**:
- Terminal window
- Command: `victor chat`
- Show TUI interface
- Type: "Hello, what can you do?"
- Show Victor's response

**Narration**:
```
Now let's try Victor for the first time. Start the interactive chat mode
by typing 'victor chat'.

[Type: victor chat]

This opens Victor's terminal UI. Let's ask Victor what it can do.

[Type: Hello, what can you do?]

Victor responds with its capabilities - code review, test generation,
refactoring, documentation, and much more. You can see the response
streaming in real-time.

Try asking Victor to review a file or add tests. It will explain what
it's doing before making any changes.
```

**Production Notes**:
- Show streaming response (don't cut)
- Highlight key parts of response
- Use zoom for small text

#### Scene 4: Provider Setup (2:30-3:30)

**Visual**:
- Split screen: Terminal + Browser
- Browser: OpenAI or Anthropic console
- Terminal: `export ANTHROPIC_API_KEY=sk-ant-...`
- Terminal: `victor chat --provider anthropic`

**Narration**:
```
By default, Victor uses local models. To use cloud providers like
Claude or GPT-4, you need to set an API key.

First, get your API key from the provider's console. For Anthropic,
go to console.anthropic.com and create a key.

[Show browser: console.anthropic.com]

Then set the environment variable in your terminal.

[Type: export ANTHROPIC_API_KEY=sk-ant-...]

Now you can use Victor with the cloud provider.

[Type: victor chat --provider anthropic]

You can also set this permanently in your config file at ~/.victor/config.yaml.
```

**Production Notes**:
- Blur actual API key in video
- Show config file example
- Mention security best practices

#### Scene 5: Common Tasks (3:30-4:30)

**Visual**:
- Show 3 terminal windows side-by-side
- Window 1: `victor chat "Review auth.py for security"`
- Window 2: `victor chat "Generate unit tests for user service"`
- Window 3: `victor chat "Add docstrings to all functions"`

**Narration**:
```
Victor excels at common coding tasks. Let's look at three examples.

First, code review. Ask Victor to review a specific file for security issues.

[Window 1: Show review in progress]

Victor analyzes the code, identifies issues, and suggests improvements.

Second, test generation. Victor can create comprehensive unit tests.

[Window 2: Show test generation]

Third, documentation. Victor adds docstrings and generates API docs.

[Window 3: Show docstring addition]

Each task completes in seconds, saving you hours of manual work.
```

**Production Notes**:
- Speed up executions (show start and end)
- Use different code snippets for variety
- Show actual file edits

#### Scene 6: Next Steps (4:30-5:00)

**Visual**:
- Title: "Next Steps"
- List of resources:
  - Quick Start Guide
  - Coordinator Guide
  - FAQ
  - GitHub Repository
- Show URL: github.com/vjsingh1984/victor

**Narration**:
```
Congratulations! You've completed your first task with Victor AI.

To learn more, check out the Quick Start Guide for a 5-minute overview,
the Coordinator Guide to understand multi-agent teams, and the FAQ for
common questions.

All documentation is available at the GitHub repository:
github.com/vjsingh1984/victor

Join our community, report issues, and contribute to the project.

Thanks for watching, and happy coding with Victor AI!
```

**Production Notes**:
- Show QR code for GitHub repo
- Display links clearly on screen
- End with Victor logo and tagline

### Alternative Formats

**YouTube Description**:
```
 Victor AI Tutorial: Get Started in 5 Minutes

Learn how to install Victor AI, set up providers, and complete your first
coding task. Victor supports 21 LLM providers including Claude, GPT-4,
and local models via Ollama.

⏱️ Timestamps:
0:00 - Introduction
0:30 - Installation
1:30 - First Chat
2:30 - Provider Setup
3:30 - Common Tasks
4:30 - Next Steps

🔗 Links:
- GitHub: https://github.com/vjsingh1984/victor
- Docs: https://github.com/vjsingh1984/victor/docs
- Quick Start: [LINK]

#VictorAI #CodingAssistant #AI #DeveloperTools
```

**Blog Post Version**:
- Convert script to written tutorial
- Include screenshots for each scene
- Add copy-paste code blocks
- Provide troubleshooting section

---

## Video 2: Multi-Agent Team Coordination (5 minutes)

**Target Audience**: Intermediate users
**Goal**: Understand and use team formations

### Script Outline

| Time | Scene | Content |
|------|-------|---------|
| 0:00-0:45 | What are Teams | Multi-agent coordination |
| 0:45-1:45 | Team Formations | 5 formation patterns |
| 1:45-2:45 | Creating Teams | Python API example |
| 2:45-3:45 | Rich Personas | Backstory and expertise |
| 3:45-4:30 | Team Memory | Persistent memory |
| 4:30-5:00 | Best Practices | When to use teams |

### Detailed Script

#### Scene 1: What are Teams (0:00-0:45)

**Visual**:
- Diagram: 3 agents working together
- Animation: Agents passing messages
- Title: "Why Use Multiple Agents?"

**Narration**:
```
Complex coding tasks benefit from multiple perspectives. Victor's team
coordination lets multiple AI agents work together using different
formations.

Think of it like a real development team. You have a security expert,
a performance specialist, and a test engineer. Each brings unique
expertise to the problem.

Victor coordinates these agents automatically, managing communication,
context sharing, and result synthesis. You just define the team and
the task - Victor handles the rest.
```

#### Scene 2: Team Formations (0:45-1:45)

**Visual**:
- Animated diagrams for each formation
- SEQUENTIAL: Chain of agents
- PARALLEL: Agents working simultaneously
- HIERARCHICAL: Manager and workers
- PIPELINE: Processing stages
- CONSENSUS: Discussion loop

**Narration**:
```
Victor supports five team formations. Let's explore each.

SEQUENTIAL is like an assembly line. Each agent works one after another,
building on previous work. Great for refinement.

PARALLEL has all agents work simultaneously. They share the same context
but work independently. Results are combined at the end. Perfect for
multiple perspectives.

HIERARCHICAL uses a manager-agent pattern. The manager delegates work
to specialists and synthesizes results. Ideal for complex tasks.

PIPELINE is a processing chain. Output from one agent feeds into the
next. Great for multi-stage workflows.

CONSENSUS requires all agents to agree. They discuss and revise until
reaching consensus. Best for critical decisions.
```

**Production Notes**:
- Use smooth animations for formations
- Color-code different agents
- Show message flow clearly

#### Scene 3: Creating Teams (1:45-2:45)

**Visual**:
- Code editor
- Show Python script creation
- Run script with output

**Narration**:
```
Let's create a team using the Python API. We'll use a pipeline formation
for a security review task.

[Show code in editor]

First, import the necessary modules from victor.teams. Create a
coordinator using the factory function.

Add team members with different roles - researcher to find vulnerabilities,
reviewer to validate findings, and executor to implement fixes.

Set the formation to PIPELINE so output flows through each stage.

Finally, execute the task with the file context.

[Run script, show output]

Victor coordinates the three agents through the pipeline, producing
a comprehensive security review with actionable fixes.
```

**Code Example**:
```python
from victor.teams import create_coordinator, TeamFormation
from victor.agent.subagents.base import SubAgentRole
from victor import Victor

vic = Victor()
coordinator = create_coordinator(vic.orchestrator)

coordinator.add_member(
    role=SubAgentRole.RESEARCHER,
    name="Security Expert",
    goal="Find vulnerabilities"
)

coordinator.add_member(
    role=SubAgentRole.REVIEWER,
    name="Code Reviewer",
    goal="Validate findings"
)

coordinator.set_formation(TeamFormation.PIPELINE)

result = await coordinator.execute_task(
    "Review auth.py for security issues",
    {"files": ["src/auth.py"]}
)
```

#### Scene 4: Rich Personas (2:45-3:45)

**Visual**:
- Split screen: Code + Result
- Show persona configuration
- Show agent's specialized behavior

**Narration**:
```
Victor supports rich personas for specialized behavior. You can define
backstory, expertise, and personality for each agent.

[Show persona configuration]

This security expert has 15 years of experience, specific expertise in
OAuth and JWT, and a methodical personality. When Victor generates the
system prompt, it incorporates all these attributes.

The agent behaves like a real security expert - thorough, focused on
security risks, and communicating with severity ratings.

This is powerful for domain-specific tasks where you want agents to
have real-world expertise and communication styles.
```

**Production Notes**:
- Show before/after comparison
- Highlight different agent behaviors
- Use concrete example

#### Scene 5: Team Memory (3:45-4:30)

**Visual**:
- Diagram: Memory storage across sessions
- Show memory configuration
- Show recall in action

**Narration**:
```
Agents can remember across tasks using persistent memory. Enable memory
and configure what to store - entities, semantic knowledge, or episodic
memories.

[Show memory config]

When an agent makes a discovery, it stores it in memory. Later tasks
can recall relevant memories automatically.

This is perfect for long-running projects where you want agents to
learn from previous work. The security expert remembers previous
vulnerabilities found. The performance specialist remembers past
optimizations.

Memory is configurable per agent, so you control what persists and
what's discarded.
```

#### Scene 6: Best Practices (4:30-5:00)

**Visual**:
- Checklist: When to use teams
- Summary of formations
- Links to learn more

**Narration**:
```
When should you use teams? Use them for complex reviews, multi-stage
tasks, cross-domain work, and quality assurance. Use single agents
for simple tasks and quick questions.

Choose the right formation - SEQUENTIAL for refinement, PARALLEL for
perspectives, HIERARCHICAL for decomposition, PIPELINE for workflows,
CONSENSUS for critical decisions.

Set realistic budgets and enable memory for long projects.

Check the Coordinator Guide for detailed documentation, and join our
community to share your team configurations!

Thanks for watching!
```

### Alternative Formats

**Interactive Tutorial**:
- Web-based tutorial with live code execution
- User can modify formations and see results
- Embedded code playgrounds

**Workshop Material**:
- Hands-on exercises
- Sample team configurations
- Group discussion questions

---

## Video 3: Advanced Team Configuration (5 minutes)

**Target Audience**: Advanced users
**Goal**: Master team customization

### Script Outline

| Time | Scene | Content |
|------|-------|---------|
| 0:00-0:30 | Overview | Advanced features |
| 0:30-1:30 | Memory System | Entity, semantic, episodic |
| 1:30-2:30 | Delegation | Hierarchical delegation |
| 2:30-3:30 | Observability | Event monitoring |
| 3:30-4:15 | Custom Formations | Create your own |
| 4:15-5:00 | Performance | Optimization tips |

### Key Points

**Memory System**:
- Three types: Entity, Semantic, Episodic
- Persistence across sessions
- Relevance thresholds
- TTL configuration

**Delegation**:
- Manager-worker pattern
- Delegation depth limits
- Target restrictions
- Approval workflows

**Observability**:
- EventBus integration
- Event types and filtering
- Metrics collection
- Debugging support

**Custom Formations**:
- Extend TeamFormation enum
- Implement coordination logic
- Message routing
- Consensus building

---

## Video 4: Provider Switching and Health (5 minutes)

**Target Audience**: All users
**Goal**: Master provider management

### Script Outline

| Time | Scene | Content |
|------|-------|---------|
| 0:00-0:45 | Provider Overview | 21 providers explained |
| 0:45-1:45 | Switching Providers | Mid-conversation switching |
| 1:45-2:45 | Health Monitoring | Proactive health checks |
| 2:45-3:45 | Fallback Strategy | Automatic failover |
| 3:45-4:30 | Cost Optimization | Reduce API costs |
| 4:30-5:00 | Best Practices | Provider selection guide |

### Key Points

**Provider Categories**:
- Cloud: Capability, speed, cost
- Local: Privacy, free, flexibility
- Hybrid: Best of both worlds

**Switching**:
- CLI command: `/provider openai`
- Python API: `switch_provider()`
- Context preservation
- Capability mapping

**Health Monitoring**:
- Latency tracking
- Error rates
- Quota monitoring
- Automatic detection

**Fallback**:
- Primary/backup configuration
- Automatic switch on failure
- Retry logic
- Manual override

---

## Video 5: Troubleshooting Common Issues (5 minutes)

**Target Audience**: All users
**Goal**: Solve common problems quickly

### Script Outline

| Time | Scene | Content |
|------|-------|---------|
| 0:00-0:30 | Overview | Diagnostic approach |
| 0:30-1:15 | Installation Issues | Setup problems |
| 1:15-2:00 | Provider Issues | API keys, connectivity |
| 2:00-2:45 | Performance Issues | Slow responses |
| 2:45-3:30 | Team Issues | Coordination problems |
| 3:30-4:15 | Tool Issues | Execution failures |
| 4:15-5:00 | Getting Help | Resources and support |

### Key Points

**Diagnostic Tools**:
- `victor doctor` - Health check
- `victor config validate` - Configuration check
- `victor provider test` - Provider connectivity
- Debug mode and logging

**Common Issues**:
- Installation: PATH problems, Python version
- Provider: API keys, rate limits, timeouts
- Performance: Context size, caching, selection
- Teams: Budgets, formations, memory
- Tools: Permissions, availability, arguments

**Getting Help**:
- Documentation
- GitHub issues
- Community forums
- Email support

---

## Production Notes

### Equipment

**Minimum**:
- Microphone: USB condenser mic ($50-100)
- Screen recording: OBS Studio (free)
- Editing: DaVinci Resolve (free) or iMovie (free)

**Recommended**:
- Microphone: Shure SM7B + audio interface ($400)
- Screen recording: CleanShot X or ScreenFlow
- Editing: Final Cut Pro or Adobe Premiere

### Recording Tips

**Audio**:
- Record in quiet room
- Use pop filter
- Speak clearly and slowly
- Monitor audio levels
- Do multiple takes

**Video**:
- Use 1920x1080 resolution
- Large, readable fonts (18pt+)
- High contrast colors
- Smooth cursor movements
- Pause on important screens

**Editing**:
- Cut mistakes cleanly
- Add transitions sparingly
- Use zoom for details
- Add subtitles for key terms
- Include progress indicators

### Distribution

**Platforms**:
- YouTube (main)
- Vimeo (backup)
- PeerTube (decentralized)

**Metadata**:
- Descriptive title
- Keyword-rich description
- Timestamps in description
- Tags: #VictorAI #CodingAssistant #AI
- Custom thumbnail

**Promotion**:
- GitHub release notes
- Twitter/X announcement
- Reddit posts (r/programming, r/python)
- Hacker News (if newsworthy)
- Developer communities

---

## Companion Resources

### Slide Decks

Each video should have a companion slide deck:
- PDF format for download
- Speaker notes
- Code examples
- Diagrams
- Resource links

### Code Repositories

Create companion repos:
- github.com/vjsingh1984/victor-examples
- One folder per video
- Runnable examples
- Requirements files
- README with instructions

### Exercise Sets

Hands-on exercises:
- Beginner: Basic tasks
- Intermediate: Team coordination
- Advanced: Custom formations
- Solutions included
- Self-grading quizzes

---

## Success Metrics

**Engagement**:
- Views: 1000+ in first month
- Watch time: 70%+ average
- Likes: 50+ per video
- Comments: 20+ per video

**Learning Outcomes**:
- Post-video quiz: 80%+ pass rate
- GitHub issues: "Help" requests decrease
- Documentation pageviews: Increase
- Community contributions: Increase

**Quality**:
- Production value: Professional
- Audio clarity: 9/10 rating
- Content accuracy: 100%
- Relevance: Highly applicable

---

**Next Steps**: Record pilot episode, gather feedback, iterate on format

---

**For questions or contributions, contact: singhvjd@gmail.com**
