from agno.workflow.condition import Condition
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

from agents import coordinator_agent, general_agent

def prepare_general_agent_input(step_input: StepInput) -> StepOutput:
    original_query = step_input.input or ""
    content = f"""
    The retriever agent could not find information in the knowledge base.
    Please answer the following query using your general knowledge or tools(if required):

    Query: {original_query}
    """
    return StepOutput(content=content)

def no_info_found(step_input: StepInput) -> bool:
    output = step_input.previous_step_content or ""
    return "no information found" in output.lower()  # adapt to your output

workflow = Workflow(
    steps=[
        Step(name="coordinator_team", agent=coordinator_agent),
        Condition(
            name="fallback_to_general",
            evaluator=no_info_found,
            steps=[
                prepare_general_agent_input,
                Step(name="general_agent", agent=general_agent)
            ],
        ),
    ]
)
