from app.utils.ai_client import get_project_client
from app.utils.agent_manager import create_agent, create_thread, run_agent, cleanup
import json, re

INSTRUCTIONS = """
You are a context summarization agent. You receive:
1. The full conversation history as JSON
2. A structured request from the input agent

Your job is to:
- Add a context_summary field summarizing relevant past conversations
- Return the input agent result EXACTLY as received, with ONLY context_summary added

STRICT RULES:
- NEVER modify, replace, or overwrite extracted_details
- NEVER add fields from history into extracted_details
- If extracted_details is a list, return it as a list — do not change it
- Only add the context_summary string field

You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no backticks.
Example:
{"intent": "create_ticket", "extracted_details": {}, "context_summary": "brief summary here"}
"""

def _parse_json_safe(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"intent": "unknown", "extracted_details": {}, "raw_response": text}

def enrich_with_context(history: list, input_agent_result: dict) -> dict:
    client    = get_project_client()
    agent     = create_agent(client, "context-agent", INSTRUCTIONS)
    thread_id = create_thread(client)

    message = (
        f"Conversation history: {json.dumps(history)}\n"
        f"Input agent result: {json.dumps(input_agent_result)}"
    )

    try:
        response  = run_agent(client, agent.id, thread_id, message)
        enriched  = _parse_json_safe(response)

        # Safety guard — if context agent corrupted extracted_details, restore original
        original_details = input_agent_result.get("extracted_details")
        enriched_details = enriched.get("extracted_details")

        if isinstance(original_details, list) and not isinstance(enriched_details, list):
            enriched["extracted_details"] = original_details

        if isinstance(original_details, list) and isinstance(enriched_details, list):
            if len(enriched_details) != len(original_details):
                enriched["extracted_details"] = original_details

        return enriched
    finally:
        cleanup(client, agent.id, thread_id)