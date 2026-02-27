"""Business automation agent recipes.

These recipes focus on business tasks like document generation,
reporting, analysis, and process automation.
"""

RECIPE_CATEGORY = "agents/specialized"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "15 minutes"


async def business_requirements_agent(
    project_description: str,
    stakeholders: list[str]
):
    """Gather and document business requirements."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    stakeholders_str = "\n".join(f"- {s}" for s in stakeholders)

    result = await agent.run(
        f"""Document business requirements for:

PROJECT: {project_description}

STAKEHOLDERS:
{stakeholders_str}

Provide:
1. Business objectives
2. Functional requirements
3. Non-functional requirements
4. User stories
5. Acceptance criteria
6. Assumptions and constraints
7. Success metrics"""
    )

    return result.content


async def use_case_agent(feature_description: str):
    """Write detailed use cases."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    result = await agent.run(
        f"""Write detailed use cases for: {feature_description}

For each use case provide:
1. Use case name and ID
2. Primary actor
3. Preconditions
4. Main flow (step by step)
5. Alternative flows
6. Postconditions
7. Business rules
8. Related requirements"""
    )

    return result.content


async def user_story_agent(epic: str, user_personas: list[str]):
    """Generate user stories from epics."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.5
    )

    personas_str = "\n".join(f"- {p}" for p in user_personas)

    result = await agent.run(
        f"""Generate user stories for epic: {epic}

USER PERSONAS:
{personas_str}

For each story:
1. As a [persona]
2. I want [feature]
3. So that [benefit]
4. Acceptance criteria
5. Story points estimate
6. Priority"""
    )

    return result.content


async def test_plan_agent(requirements: str):
    """Generate test plan from requirements."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate comprehensive test plan for:

REQUIREMENTS:
{requirements}

Provide:
1. Test scope
2. Test types (unit, integration, E2E, performance)
3. Test scenarios
4. Test cases
5. Test data requirements
6. Entry/exit criteria
7. Test schedule
8. Resource requirements"""
    )

    return result.content


async def release_notes_agent(
    version: str,
    features: list[str],
    bug_fixes: list[str]
):
    """Generate release notes."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    features_str = "\n".join(f"- {f}" for f in features)
    fixes_str = "\n".join(f"- {f}" for f in bug_fixes)

    result = await agent.run(
        f"""Generate release notes for version {version}.

NEW FEATURES:
{features_str}

BUG FIXES:
{fixes_str}

Provide:
1. Release summary
2. Highlights
3. New features (with descriptions)
4. Bug fixes
5. Known issues
6. Upgrade instructions
7. Breaking changes (if any)"""
    )

    return result.content


async def incident_report_agent(incident_description: str):
    """Generate incident report."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate incident report for:

{incident_description}

Provide:
1. Incident summary
2. Impact assessment
3. Timeline of events
4. Root cause analysis
5. Resolution steps
6. Lessons learned
7. Action items to prevent recurrence
8. Follow-up requirements"""
    )

    return result.content


async def project_plan_agent(
    project_goal: str,
    duration_weeks: int,
    team_size: int
):
    """Generate project plan."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    result = await agent.run(
        f"""Generate project plan for:

GOAL: {project_goal}
DURATION: {duration_weeks} weeks
TEAM SIZE: {team_size} people

Provide:
1. Project phases
2. Milestones and deliverables
3. Work breakdown structure
4. Timeline (Gantt chart format)
5. Resource allocation
6. Dependencies
7. Risk assessment
8. Communication plan"""
    )

    return result.content


async def meeting_notes_agent(
    meeting_transcript: str,
    meeting_type: str = "team standup"
):
    """Generate structured meeting notes."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate structured meeting notes from transcript:

MEETING TYPE: {meeting_type}

TRANSCRIPT:
{meeting_transcript}

Provide:
1. Attendees
2. Agenda items with outcomes
3. Decisions made
4. Action items (with owners and due dates)
5. Discussion highlights
6. Next steps
7. Parking lot items"""
    )

    return result.content


async def competitive_analysis_agent(company: str, competitors: list[str]):
    """Analyze competitive landscape."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search"],
        temperature=0.3
    )

    competitors_str = "\n".join(f"- {c}" for c in competitors)

    result = await agent.run(
        f"""Perform competitive analysis for:

COMPANY: {company}

COMPETITORS:
{competitors_str}

For each competitor:
1. Product/service comparison
2. Pricing analysis
3. Market positioning
4. Strengths and weaknesses
5. Recent news/developments

Provide:
1. Competitive matrix
2. Market share overview
3. Differentiation opportunities
4. Threats assessment"""
    )

    return result.content


async def market_research_agent(market: str, focus_areas: list[str]):
    """Conduct market research."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search"],
        temperature=0.3
    )

    focus_str = "\n".join(f"- {f}" for f in focus_areas)

    result = await agent.run(
        f"""Conduct market research for:

MARKET: {market}

FOCUS AREAS:
{focus_str}

Provide:
1. Market size and growth
2. Market trends
3. Customer segments
4. Pain points
5. Competitive landscape
6. Opportunities
7. Threats
8. Recommendations"""
    )

    return result.content


async def pricing_strategy_agent(
    product: str,
    costs: str,
    target_market: str
):
    """Develop pricing strategy."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    result = await agent.run(
        f"""Develop pricing strategy for:

PRODUCT: {product}
COSTS: {costs}
TARGET MARKET: {target_market}

Provide:
1. Pricing objectives
2. Pricing models (cost-plus, value-based, competitive)
3. Recommended price points
4. Tier options (if applicable)
5. Discount strategy
6. Price testing approach
7. Competitive comparison"""
    )

    return result.content


async def go_to_market_agent(
    product: str,
    launch_date: str,
    budget: str
):
    """Create go-to-market plan."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    result = await agent.run(
        f"""Create go-to-market plan for:

PRODUCT: {product}
LAUNCH DATE: {launch_date}
BUDGET: {budget}

Provide:
1. Target audience definition
2. Value proposition
3. Messaging framework
4. Marketing channels
5. Sales strategy
6. Launch timeline
7. KPIs and metrics
8. Budget allocation"""
    )

    return result.content


async def financial_analysis_agent(
    business_type: str,
    metrics: list[str]
):
    """Analyze financial metrics and projections."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    metrics_str = "\n".join(f"- {m}" for m in metrics)

    result = await agent.run(
        f"""Perform financial analysis for:

BUSINESS TYPE: {business_type}

METRICS TO ANALYZE:
{metrics_str}

Provide:
1. Revenue projections
2. Cost structure analysis
3. Break-even analysis
4. Cash flow forecast
5. Key financial ratios
6. Sensitivity analysis
7. Scenario planning
8. Recommendations"""
    )

    return result.content


async def business_intelligence_agent(
    data_description: str,
    business_questions: list[str]
):
    """Generate BI insights from data."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    questions_str = "\n".join(f"- {q}" for q in business_questions)

    result = await agent.run(
        f"""Generate business intelligence insights:

DATA: {data_description}

BUSINESS QUESTIONS:
{questions_str}

Provide:
1. Key findings
2. Trends and patterns
3. Anomalies
4. Correlations
5. Recommendations
6. Visualizations to create
7. Python analysis code"""
    )

    return result.content


async def contract_review_agent(contract_text: str):
    """Review contract for key terms and risks."""
    from victor import Agent

    agent = Agent.create(
        vertical="security",
        temperature=0.2
    )

    result = await agent.run(
        f"""Review this contract and identify key terms and risks:

{contract_text}

Provide:
1. Key terms summary
2. Obligations of each party
3. Payment terms
4. Termination clauses
5. Liability limitations
6. Risk areas
7. Missing clauses
8. Recommendations for revision"""
    )

    return result.content


async def rfp_response_agent(
    rfp_document: str,
    company_capabilities: str
):
    """Generate RFP response."""
    from victor import Agent

    agent = Agent.create(
        temperature=0.4
    )

    result = await agent.run(
        f"""Generate RFP response based on:

RFP REQUIREMENTS:
{rfp_document}

OUR CAPABILITIES:
{company_capabilities}

Provide:
1. Executive summary
2. Understanding of requirements
3. Proposed solution
4. Implementation approach
5. Timeline
6. Pricing structure
7. Team qualifications
8. Differentiators"""
    )

    return result.content


async def demo_business_agents():
    """Demonstrate business agent recipes."""
    print("=== Business Agent Recipes ===\n")

    print("1. Business Requirements:")
    result = await business_requirements_agent(
        "Build customer support portal",
        ["Product Manager", "Support Team", "Customers"]
    )
    print(result[:300] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_business_agents())
