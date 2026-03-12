from azure.ai.agents.models import FunctionTool, ToolSet, MessageRole
from app.utils.ai_client import get_project_client
from app.utils.agent_manager import create_thread, cleanup
from app.config import Config
import json

INSTRUCTIONS = """
You are an operations agent with tools to create, update, search, 
get, transition, comment on, and link tickets or work items.

Based on the enriched request, call the appropriate tool with the 
correct parameters. Always use the project details from the request.
Return the raw tool result.
"""


def execute_jira_task(enriched_request: dict, platform: str = "jira") -> dict:
    """
    Execute a task on either Jira or Azure DevOps based on platform parameter.
    platform: "jira" or "devops"
    """

    # Load the correct toolset based on platform
    if platform == "devops":
        from app.agents.devops_tools import devops_functions as functions
    else:
        from app.agents.jira_tools import jira_functions as functions

    client    = get_project_client()
    thread_id = create_thread(client)

    tool    = FunctionTool(functions=functions)
    toolset = ToolSet()
    toolset.add(tool)
    client.agents.enable_auto_function_calls(toolset)

    agent = client.agents.create_agent(
        model        = Config.AZURE_OPENAI_DEPLOYMENT,
        name         = f"{platform}-agent",
        instructions = INSTRUCTIONS,
        toolset      = toolset
    )

    try:
        client.agents.messages.create(
            thread_id = thread_id,
            role      = MessageRole.USER,
            content   = f"Execute this request on {platform}: {json.dumps(enriched_request)}"
        )

        run = client.agents.runs.create_and_process(
            thread_id = thread_id,
            agent_id  = agent.id
        )

        if run.status == "failed":
            return {"status": "error", "detail": f"Agent run failed: {run.last_error}"}

        messages = client.agents.messages.list(thread_id=thread_id)
        for msg in messages:
            if msg.role == MessageRole.AGENT:
                for block in msg.content:
                    if hasattr(block, "text"):
                        try:
                            return json.loads(block.text.value)
                        except json.JSONDecodeError:
                            return {"status": "success", "response": block.text.value}

        return {"status": "error", "detail": "No response from agent"}

    finally:
        cleanup(client, agent.id, thread_id)