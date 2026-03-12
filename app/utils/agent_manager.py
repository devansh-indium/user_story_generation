from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageRole, ListSortOrder
from app.config import Config


def create_agent(client: AIProjectClient, name: str, instructions: str):
    """Creates a fresh Foundry Agent and returns it."""
    return client.agents.create_agent(
        model        = Config.AZURE_OPENAI_DEPLOYMENT,
        name         = name,
        instructions = instructions
    )


def create_thread(client: AIProjectClient) -> str:
    """Creates a new conversation thread and returns its ID."""
    thread = client.agents.threads.create()
    return thread.id


def run_agent(client: AIProjectClient, agent_id: str, thread_id: str, user_message: str) -> str:
    """
    Posts a message to a thread, runs the agent,
    waits for completion and returns the response text.
    """
    # Post message to thread
    client.agents.messages.create(
        thread_id = thread_id,
        role      = MessageRole.USER,
        content   = user_message
    )

    # Run agent and wait for completion (blocking)
    run = client.agents.runs.create_and_process(
        thread_id = thread_id,
        agent_id  = agent_id
    )

    if run.status == "failed":
        return f"Agent run failed: {run.last_error}"

    # Get latest assistant message
    messages = client.agents.messages.list(
        thread_id = thread_id,
        order     = ListSortOrder.DESCENDING
    )

    for msg in messages:
        if msg.role == MessageRole.AGENT:
            for block in msg.content:
                if hasattr(block, "text"):
                    return block.text.value

    return ""


def cleanup(client: AIProjectClient, agent_id: str, thread_id: str):
    """Delete agent and thread after use to keep things clean."""
    try:
        client.agents.delete_agent(agent_id)
        client.agents.threads.delete(thread_id)
    except Exception:
        pass