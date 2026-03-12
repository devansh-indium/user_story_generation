from app.utils.ai_client import get_project_client
from app.utils.agent_manager import create_agent, create_thread, run_agent, cleanup
import json

INSTRUCTIONS = """
You are a friendly response agent. You receive the result of a Jira operation.
Explain what happened in a clear, friendly, conversational way.
Be concise and helpful. If there was an error, explain it simply and suggest what to do.
"""

def generate_response(jira_result: dict, original_message: str) -> str:
    client    = get_project_client()
    agent     = create_agent(client, "response-agent", INSTRUCTIONS)
    thread_id = create_thread(client)

    message = (
        f"User asked: {original_message}\n"
        f"Jira result: {json.dumps(jira_result)}"
    )

    try:
        response = run_agent(client, agent.id, thread_id, message)
        return response
    finally:
        cleanup(client, agent.id, thread_id)