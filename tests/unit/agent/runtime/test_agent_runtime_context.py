from victor.agent.runtime.context import AgentRuntimeContext


def test_agent_runtime_context_identity_metadata():
    context = AgentRuntimeContext(
        agent_id="agent_researcher_1",
        display_name="Researcher",
        role="researcher",
        session_id="session_child",
        parent_session_id="session_root",
        team_id="team_1",
        plan_id="plan_1",
        plan_step_id="1",
        member_id="step_1_researcher",
    )

    assert context.identity_metadata() == {
        "agent_id": "agent_researcher_1",
        "display_name": "Researcher",
        "role": "researcher",
        "session_id": "session_child",
        "parent_session_id": "session_root",
        "team_id": "team_1",
        "plan_id": "plan_1",
        "plan_step_id": "1",
        "member_id": "step_1_researcher",
    }


def test_agent_runtime_context_child_session_derivation():
    root = AgentRuntimeContext(
        agent_id="root_agent",
        display_name="Root Agent",
        role="manager",
        session_id="session_root",
    )

    child = root.derive_child(
        agent_id="team_1_step_1",
        display_name="Step 1 Researcher",
        role="researcher",
        member_id="step_1_researcher",
        team_id="team_1",
        plan_id="plan_1",
        plan_step_id="1",
    )

    assert child.parent_session_id == "session_root"
    assert child.session_id == "session_root:team_1:step_1_researcher"
    assert child.agent_id == "team_1_step_1"
