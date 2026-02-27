"""Specialized research agent recipes.

These recipes focus on research tasks like literature review,
fact-checking, summarization, and academic writing assistance.
"""

RECIPE_CATEGORY = "agents/specialized"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "15 minutes"


async def literature_review_agent(topic: str, num_papers: int = 10):
    """Conduct literature review on a topic."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Conduct a literature review on: {topic}

Find and analyze {num_papers} recent papers.

Provide:
1. Key themes and trends
2. Main methodologies used
3. Important findings
4. Research gaps
5. Future directions
6. Key papers with citations
7. Bibliography in APA format"""
    )

    return result.content


async def fact_check_agent(claims: list[str]):
    """Fact-check claims against reliable sources."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.2
    )

    claims_str = "\n".join(f"- {c}" for c in claims)

    result = await agent.run(
        f"""Fact-check these claims:

{claims_str}

For each claim provide:
1. Verdict (True/False/Mixed/Unverifiable)
2. Supporting evidence
3. Sources with links
4. Context and nuance
5. Confidence level"""
    )

    return result.content


async def citation_verification_agent(text: str):
    """Verify citations in academic text."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.2
    )

    result = await agent.run(
        f"""Verify all citations in this text:

{text}

For each citation:
1. Check if source exists
2. Verify accuracy of引用
3. Check if claims are supported
4. Identify any misquotations
5. Flag suspicious or unreliable sources"""
    )

    return result.content


async def academic_writing_agent(
    topic: str,
    paper_type: str = "review",
    target_venue: str = ""
):
    """Assist with academic writing."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.5
    )

    result = await agent.run(
        f"""Help write a {paper_type} paper on: {topic}

Target venue: {target_venue if target_venue else 'General academic venue'}

Provide:
1. Suggested title
2. Abstract (150-250 words)
3. Introduction outline
4. Section structure
5. Key arguments per section
6. Conclusion outline
7. Suggested keywords"""
    )

    return result.content


async def plagiarism_check_agent(text: str):
    """Check text for potential plagiarism issues."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search"],
        temperature=0.2
    )

    result = await agent.run(
        f"""Check this text for plagiarism:

{text}

For each significant section:
1. Search for similar content online
2. Identify exact matches
3. Identify paraphrased content
4. Flag problematic sections
5. Suggest proper citations
6. Provide originality score"""
    )

    return result.content


async def research_methodology_agent(research_question: str):
    """Design research methodology."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Design research methodology for:

RESEARCH QUESTION: {research_question}

Provide:
1. Research approach (quantitative/qualitative/mixed)
2. Data collection methods
3. Sampling strategy
4. Analysis approach
5. Validity and reliability measures
6. Ethical considerations
7. Timeline and resources needed"""
    )

    return result.content


async def survey_design_agent(research_objectives: str):
    """Design survey questionnaire."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Design a survey for:

{research_objectives}

Provide:
1. Survey sections and flow
2. Question wording
3. Response scales
4. Skip logic
5. Demographic questions
6. Pilot testing recommendations
7. Analysis plan"""
    )

    return result.content


async def interview_protocol_agent(research_topic: str):
    """Design interview protocol."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Design interview protocol for: {research_topic}

Provide:
1. Interview guide with questions
2. Probing techniques
3. Opening and closing
4. Consent script
5. Note-taking template
6. Recording procedures
7. Analysis approach"""
    )

    return result.content


async def data_analysis_plan_agent(research_questions: list[str], data_type: str):
    """Create data analysis plan."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    questions_str = "\n".join(f"- {q}" for q in research_questions)

    result = await agent.run(
        f"""Create data analysis plan for:

RESEARCH QUESTIONS:
{questions_str}

DATA TYPE: {data_type}

Provide:
1. Data preprocessing steps
2. Analysis methods for each question
3. Statistical tests
4. Visualization approach
5. Software/tools needed
6. Validation approach
7. Reporting format"""
    )

    return result.content


async def grant_proposal_agent(
    project_description: str,
    funding_agency: str
):
    """Assist with grant proposal writing."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Help write grant proposal for:

PROJECT: {project_description}

AGENCY: {funding_agency}

Provide:
1. Project summary
2. Specific aims
3. Significance and innovation
4. Approach outline
5. Evaluation plan
6. Budget categories
7. Timeline"""
    )

    return result.content


async def systematic_review_agent(
    research_question: str,
    databases: list[str]
):
    """Design systematic review protocol."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        tools=["web_search"],
        temperature=0.3
    )

    dbs = ", ".join(databases)

    result = await agent.run(
        f"""Design systematic review protocol for:

RESEARCH QUESTION: {research_question}

DATABASES: {dbs}

Provide:
1. PICO framework breakdown
2. Search strategy for each database
3. Inclusion/exclusion criteria
4. Quality assessment tools
5. Data extraction form
6. Synthesis approach
7. Risk of bias assessment"""
    )

    return result.content


async def meta_analysis_agent(studies: list[dict]):
    """Design meta-analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design meta-analysis for {len(studies)} studies.

Provide:
1. Effect size metrics
2. Statistical models
3. Heterogeneity assessment
4. Publication bias analysis
5. Sensitivity analysis
6. Forest plot approach
7. Software recommendations"""
    )

    return result.content


async def conference_abstract_agent(
    research_summary: str,
    conference: str
):
    """Write conference abstract."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Write conference abstract for:

CONFERENCE: {conference}

RESEARCH SUMMARY:
{research_summary}

Provide:
1. Title (attention-grabbing)
2. Abstract (250 words max):
   - Background
   - Methods
   - Results
   - Conclusion
3. Keywords
4. Presentation format suggestions"""
    )

    return result.content


async def poster_design_agent(research_content: str):
    """Design academic poster."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Design academic poster for:

{research_content}

Provide:
1. Layout recommendation
2. Section organization
3. Visual hierarchy
4. Key figures and tables
5. Color scheme
6. Font recommendations
7. Content for each section"""
    )

    return result.content


async def presentation_script_agent(
    slides_content: str,
    duration_minutes: int
):
    """Write presentation script."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.4
    )

    result = await agent.run(
        f"""Write {duration_minutes}-minute presentation script for:

{slides_content}

Provide:
1. Opening hook
2. Section transitions
3. Slide-by-slide script
4. Timing cues
5. Emphasis points
6. Closing statement
7. Q&A preparation"""
    )

    return result.content


async def peer_review_agent(paper_content: str):
    """Conduct peer review of paper."""
    from victor import Agent

    agent = Agent.create(
        vertical="research",
        temperature=0.3
    )

    result = await agent.run(
        f"""Conduct peer review of:

{paper_content}

Provide:
1. Overall assessment
2. Major comments
3. Minor suggestions
4. Strengths of the paper
5. Areas for improvement
6. Recommended decision (Accept/Minor Revision/Major Revision/Reject)
7. Comments for author"""
    )

    return result.content


async def demo_research_agents():
    """Demonstrate research agent recipes."""
    print("=== Research Agent Recipes ===\n")

    print("1. Literature Review:")
    result = await literature_review_agent("machine learning interpretability", 5)
    print(result[:300] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_research_agents())
