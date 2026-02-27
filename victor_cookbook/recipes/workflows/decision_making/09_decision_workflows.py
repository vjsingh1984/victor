"""Decision-making workflow recipes.

These workflows help with structured decision-making processes,
consensus building, and evaluation frameworks.
"""

RECIPE_CATEGORY = "workflows/decision_making"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "20 minutes"


async def swot_analysis_workflow(topic: str):
    """Structured SWOT analysis workflow."""
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "SWOT Analysis"
description: "Structured Strengths, Weaknesses, Opportunities, Threats analysis"

nodes:
  - id: "identify_strengths"
    type: "agent"
    config:
      prompt: |
        Identify Strengths for: {{input.topic}}
        Consider internal advantages, resources, capabilities.
        Be specific and actionable.

  - id: "identify_weaknesses"
    type: "agent"
    config:
      prompt: |
        Identify Weaknesses for: {{input.topic}}
        Consider internal limitations, gaps, areas for improvement.
        Be honest and constructive.

  - id: "identify_opportunities"
    type: "agent"
    config:
      prompt: |
        Identify Opportunities for: {{input.topic}}
        Consider external trends, market conditions, potential growth areas.
        Think strategically and long-term.

  - id: "identify_threats"
    type: "agent"
    config:
      prompt: |
        Identify Threats for: {{input.topic}}
        Consider external risks, competition, market challenges.
        Be realistic and prepare contingencies.

  - id: "prioritize_findings"
    type: "agent"
    config:
      prompt: |
        Prioritize and synthesize SWOT findings:
        Strengths: {{identify_strengths.output}}
        Weaknesses: {{identify_weaknesses.output}}
        Opportunities: {{identify_opportunities.output}}
        Threats: {{identify_threats.output}}

        For each category, rank items by impact and urgency.
        Identify top 3-5 in each category.

  - id: "generate_strategy"
    type: "agent"
    config:
      prompt: |
        Generate strategic recommendations based on prioritized SWOT:
        {{prioritize_findings.output}}

        Create:
        1. SO strategies (leverage strengths for opportunities)
        2. WO strategies (address weaknesses to pursue opportunities)
        3. ST strategies (use strengths to mitigate threats)
        4. WT strategies (minimize weaknesses to avoid threats)

  - id: "action_plan"
    type: "agent"
    config:
      prompt: |
        Create action plan based on SWOT strategies:
        {{generate_strategy.output}}

        For each strategy:
        - Specific actions
        - Timeline
        - Resources needed
        - Success metrics
        - Responsible parties

edges:
  - from: "start"
    to: "identify_strengths"
  - from: "start"
    to: "identify_weaknesses"
  - from: "start"
    to: "identify_opportunities"
  - from: "start"
    to: "identify_threats"
  - from: "identify_strengths"
    to: "prioritize_findings"
  - from: "identify_weaknesses"
    to: "prioritize_findings"
  - from: "identify_opportunities"
    to: "prioritize_findings"
  - from: "identify_threats"
    to: "prioritize_findings"
  - from: "prioritize_findings"
    to: "generate_strategy"
  - from: "generate_strategy"
    to: "action_plan"
  - from: "action_plan"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"topic": topic}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def pros_cons_workflow(decision: str, criteria: list[str]):
    """Structured pros/cons analysis workflow."""
    import tempfile
    import os
    from victor import Agent

    criteria_str = "\n".join(f"- {c}" for c in criteria)

    workflow_yaml = f"""
name: "Pros and Cons Analysis"
description: "Structured evaluation of advantages and disadvantages"

nodes:
  - id: "identify_pros"
    type: "agent"
    config:
      prompt: |
        Identify all potential advantages (pros) for:
        {{input.decision}}

        Consider these criteria:
        {criteria_str}

        Be thorough and creative.

  - id: "identify_cons"
    type: "agent"
    config:
      prompt: |
        Identify all potential disadvantages (cons) for:
        {{input.decision}}

        Consider these criteria:
        {criteria_str}

        Be realistic and thorough.

  - id: "assess_impact"
    type: "agent"
    config:
      prompt: |
        Assess the impact and likelihood of each pro and con:
        Pros: {{identify_pros.output}}
        Cons: {{identify_cons.output}}

        For each item:
        - Impact score (1-10)
        - Likelihood score (1-10)
        - Time horizon (short/medium/long term)

  - id: "weigh_analysis"
    type: "agent"
    config:
      prompt: |
        Weigh the pros and cons based on impact assessment:
        {{assess_impact.output}}

        Calculate:
        - Total weighted pros score
        - Total weighted cons score
        - Net score
        - Risk/reward ratio

  - id: "recommendation"
    type: "agent"
    config:
      prompt: |
        Based on the weighted analysis:
        {{weigh_analysis.output}}

        Provide:
        1. Clear recommendation (proceed/avoid/proceed with modifications)
        2. Confidence level
        3. Key factors driving the decision
        4. Mitigation strategies for major cons
        5. Success criteria

edges:
  - from: "start"
    to: "identify_pros"
  - from: "start"
    to: "identify_cons"
  - from: "identify_pros"
    to: "assess_impact"
  - from: "identify_cons"
    to: "assess_impact"
  - from: "assess_impact"
    to: "weigh_analysis"
  - from: "weigh_analysis"
    to: "recommendation"
  - from: "recommendation"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"decision": decision}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def multi_criteria_decision_workflow(
    decision: str,
    options: list[str],
    criteria: list[str],
    weights: list[float]
):
    """Multi-criteria decision analysis workflow."""
    import tempfile
    import os
    from victor import Agent

    options_str = "\n".join(f"- {o}" for o in options)
    criteria_str = "\n".join(f"- {c}: {w}" for c, w in zip(criteria, weights))

    workflow_yaml = f"""
name: "Multi-Criteria Decision Analysis"
description: "Weighted scoring of multiple options against criteria"

nodes:
  - id: "score_options"
    type: "agent"
    config:
      prompt: |
        Score each option against each criteria (1-10 scale):

        DECISION: {{input.decision}}

        OPTIONS:
        {options_str}

        CRITERIA (with weights):
        {criteria_str}

        Create a scoring matrix.

  - id: "calculate_weighted_scores"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Calculate weighted scores:
        {{score_options.output}}

        For each option:
        1. Multiply score by weight for each criterion
        2. Sum the weighted scores
        3. Normalize to 0-100 scale

        Provide rankings.

  - id: "sensitivity_analysis"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Perform sensitivity analysis on the weighted scores:
        {{calculate_weighted_scores.output}}

        Test:
        1. What if top criterion weight changes ±20%?
        2. What if lowest-scoring option improves in key areas?
        3. Identify any close calls that could flip with small changes

  - id: "final_recommendation"
    type: "agent"
    config:
      prompt: |
        Based on MCDA and sensitivity analysis:
        {{calculate_weighted_scores.output}}
        {{sensitivity_analysis.output}}

        Provide:
        1. Recommended option
        2. Confidence level
        3. Key differentiators
        4. Conditions that would change recommendation
        5. Implementation considerations

edges:
  - from: "start"
    to: "score_options"
  - from: "score_options"
    to: "calculate_weighted_scores"
  - from: "calculate_weighted_scores"
    to: "sensitivity_analysis"
  - from: "sensitivity_analysis"
    to: "final_recommendation"
  - from: "final_recommendation"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"decision": decision}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def risk_assessment_workflow(project: str):
    """Risk assessment and mitigation workflow."""
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "Risk Assessment"
description: "Identify, assess, and plan for risks"

nodes:
  - id: "identify_risks"
    type: "agent"
    config:
      prompt: |
        Identify potential risks for: {{input.project}}

        Risk categories:
        - Technical risks
        - Financial risks
        - Operational risks
        - Schedule risks
        - Legal/compliance risks
        - Reputational risks

        For each risk: describe clearly and specifically.

  - id: "assess_likelihood"
    type: "agent"
    config:
      prompt: |
        Assess likelihood of each risk (1-5 scale):
        {{identify_risks.output}}

        1 = Very unlikely
        2 = Unlikely
        3 = Possible
        4 = Likely
        5 = Very likely

        Provide rationale for each assessment.

  - id: "assess_impact"
    type: "agent"
    config:
      prompt: |
        Assess impact of each risk if it occurs (1-5 scale):
        {{identify_risks.output}}

        1 = Negligible
        2 = Minor
        3 = Moderate
        4 = Major
        5 = Severe/catastrophic

        Consider cost, time, quality, reputation.

  - id: "calculate_risk_scores"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Calculate risk scores:
        Likelihood: {{assess_likelihood.output}}
        Impact: {{assess_impact.output}}

        Risk Score = Likelihood × Impact

        Create risk matrix:
        - High risks (score 15-25)
        - Medium risks (score 8-14)
        - Low risks (score 1-7)

  - id: "develop_mitigations"
    type: "agent"
    config:
      prompt: |
        Develop mitigation strategies for high and medium risks:
        {{calculate_risk_scores.output}}

        For each risk:
        1. Prevention strategies (reduce likelihood)
        2. Contingency strategies (reduce impact)
        3. Early warning indicators
        4. Owner/responsible party

  - id: "create_monitoring_plan"
    type: "agent"
    config:
      prompt: |
        Create risk monitoring plan:
        {{develop_mitigations.output}}

        Specify:
        1. What to monitor for each risk
        2. Frequency of monitoring
        3. Reporting format
        4. Escalation triggers
        5. Review schedule

edges:
  - from: "start"
    to: "identify_risks"
  - from: "identify_risks"
    to: "assess_likelihood"
  - from: "identify_risks"
    to: "assess_impact"
  - from: "assess_likelihood"
    to: "calculate_risk_scores"
  - from: "assess_impact"
    to: "calculate_risk_scores"
  - from: "calculate_risk_scores"
    to: "develop_mitigations"
  - from: "develop_mitigations"
    to: "create_monitoring_plan"
  - from: "create_monitoring_plan"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"project": project}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def root_cause_analysis_workflow(problem: str):
    """Root cause analysis workflow using 5 Whys and fishbone."""
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "Root Cause Analysis"
description: "Identify root causes using multiple techniques"

nodes:
  - id: "define_problem"
    type: "agent"
    config:
      prompt: |
        Clearly define the problem:
        {{input.problem}}

        Provide:
        1. Problem statement (specific, measurable)
        2. Symptoms observed
        3. Impact/current state
        4. Desired state

  - id: "five_whys"
    type: "agent"
    config:
      prompt: |
        Apply 5 Whys technique to:
        {{define_problem.output}}

        Ask "why" 5 times to drill down to root cause.
        Document each why and the answer.

  - id: "fishbone_analysis"
    type: "agent"
    config:
      prompt: |
        Create fishbone (Ishikawa) diagram for:
        {{define_problem.output}}

        Categories to explore:
        - People
        - Process
        - Equipment
        - Materials
        - Environment
        - Management

        Identify potential causes in each category.

  - id: "synthesize_causes"
    type: "agent"
    config:
      prompt: |
        Synthesize findings from both techniques:
        5 Whys: {{five_whys.output}}
        Fishbone: {{fishbone_analysis.output}}

        Identify:
        1. Root causes (consensus from both methods)
        2. Contributing factors
        3. Correlations vs causations

  - id: "verify_causes"
    type: "agent"
    config:
      prompt: |
        Design verification approach for root causes:
        {{synthesize_causes.output}}

        For each potential root cause:
        1. How to verify it's actually a cause
        2. Data/evidence needed
        3. Testing approach
        4. Timeline for verification

  - id: "solutions"
    type: "agent"
    config:
      prompt: |
        Generate solutions addressing verified root causes:
        {{verify_causes.output}}

        For each root cause:
        1. Solution options
        2. Feasibility assessment
        3. Expected effectiveness
        4. Implementation requirements
        5. Success metrics

edges:
  - from: "start"
    to: "define_problem"
  - from: "define_problem"
    to: "five_whys"
  - from: "define_problem"
    to: "fishbone_analysis"
  - from: "five_whys"
    to: "synthesize_causes"
  - from: "fishbone_analysis"
    to: "synthesize_causes"
  - from: "synthesize_causes"
    to: "verify_causes"
  - from: "verify_causes"
    to: "solutions"
  - from: "solutions"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"problem": problem}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def demo_decision_workflows():
    """Demonstrate decision-making workflows."""
    print("=== Decision-Making Workflow Recipes ===\n")

    print("1. SWOT Analysis:")
    result = await swot_analysis_workflow("Launch new SaaS product")
    print(result[:400] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_decision_workflows())
