from flask import Blueprint, request, jsonify, session
from app import db
from app.models.conversation import Conversation, ConversationEmbedding
from app.agents.input_agent import process_input
from app.agents.context_agent import enrich_with_context
from app.agents.jira_agent import execute_jira_task
from app.agents.response_agent import generate_response
from app.utils.file_reader import extract_text, truncate_text, is_image
from app.utils.image_analyser import analyse_image
from app.utils.embedder import get_embedding
import uuid
import json

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    # ---------------------------------------------------------------------------
    # Parse request — supports both JSON and multipart/form-data (file uploads)
    # ---------------------------------------------------------------------------
    if request.content_type and "multipart/form-data" in request.content_type:
        message  = request.form.get("message", "")
        platform = request.form.get("platform", "jira")
        file     = request.files.get("file")
    else:
        data     = request.get_json()
        message  = data.get("message", "")
        platform = data.get("platform", "jira")
        file     = None

    # Validate
    if not message:
        return jsonify({"error": "message is required"}), 400

    if platform not in ("jira", "devops"):
        return jsonify({"error": "platform must be 'jira' or 'devops'"}), 400

    # ---------------------------------------------------------------------------
    # Handle uploaded file — image or document
    # ---------------------------------------------------------------------------
    file_content  = None
    file_error    = None
    file_filename = None
    file_type     = None   # "image" or "document"

    if file and file.filename:
        file_filename = file.filename
        try:
            if is_image(file):
                # --- IMAGE: analyse with GPT-4o vision ---
                file_type    = "image"
                analysis     = analyse_image(file, context=message)
                file_content = analysis
                print(f"Image analysed: {file_filename} ({len(analysis)} chars)")

            else:
                # --- DOCUMENT: extract text ---
                file_type    = "document"
                raw_text     = extract_text(file)
                file_content = truncate_text(raw_text, max_chars=8000)
                print(f"Document extracted: {file_filename} ({len(file_content)} chars)")

        except (ValueError, ImportError, RuntimeError) as e:
            file_error = str(e)
            print(f"File processing error: {file_error}")

    # ---------------------------------------------------------------------------
    # Build the full message for the input agent
    # ---------------------------------------------------------------------------
    if file_content and file_type == "image":
        full_message = (
            f"{message}\n\n"
            f"--- ATTACHED IMAGE ANALYSIS: {file_filename} ---\n"
            f"{file_content}\n"
            f"--- END OF IMAGE ANALYSIS ---\n\n"
            f"Use the image analysis above to populate the ticket description, "
            f"steps to reproduce, environment details, and any other relevant fields."
        )
    elif file_content and file_type == "document":
        full_message = (
            f"{message}\n\n"
            f"--- ATTACHED DOCUMENT: {file_filename} ---\n"
            f"{file_content}\n"
            f"--- END OF DOCUMENT ---"
        )
    else:
        full_message = message

    # Logging
    print(f"Received message: {message} | Platform: {platform}")
    if file_filename:
        print(f"With attached {file_type}: {file_filename}")
    if file_error:
        print(f"File error: {file_error}")

    # ---------------------------------------------------------------------------
    # Session management
    # ---------------------------------------------------------------------------
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]

    # ---------------------------------------------------------------------------
    # Load conversation history
    # ---------------------------------------------------------------------------
    history_rows = Conversation.query.filter_by(
        session_id=session_id
    ).order_by(Conversation.created_at).all()
    history = [row.to_dict() for row in history_rows]

    # ---------------------------------------------------------------------------
    # Run agent pipeline
    # ---------------------------------------------------------------------------
    print("Processing input...")
    input_result = process_input(full_message)
    print(f"Input agent result: {input_result}")

    # Pass session_id and user_message so context agent can do RAG retrieval
    enriched = enrich_with_context(
        history,
        input_result,
        session_id=session_id,
        user_message=message
    )
    print(f"Enriched request: {enriched}")

    # Handle bulk ticket creation (list of tickets from document)
    details = enriched.get("extracted_details", {})

    if isinstance(details, list):
        results = []
        for ticket in details:
            single = {**enriched, "extracted_details": ticket}
            result = execute_jira_task(single, platform=platform)
            results.append(result)
            print(f"Created ticket: {result}")
        task_result = {
            "status":  "success",
            "bulk":    True,
            "results": results,
            "summary": f"{sum(1 for r in results if r.get('status') == 'success')} tickets created."
        }
    else:
        task_result = execute_jira_task(enriched, platform=platform)

    print(f"Task execution result: {task_result}")

    final_response = generate_response(task_result, message)
    print(f"Final response: {final_response}")

    # ---------------------------------------------------------------------------
    # Save conversation to database
    # ---------------------------------------------------------------------------
    user_content = message
    if file_filename:
        user_content += f" [Attached {file_type}: {file_filename}]"

    db.session.add(Conversation(session_id=session_id, role="user",      content=user_content))
    db.session.add(Conversation(session_id=session_id, role="assistant", content=final_response))
    db.session.commit()

    # ---------------------------------------------------------------------------
    # Store embeddings for RAG retrieval in future conversations
    # ---------------------------------------------------------------------------
    try:
        if isinstance(details, list):
            # Bulk creation — embed each ticket individually
            for i, ticket in enumerate(details):
                ticket_result = task_result["results"][i] if i < len(task_result.get("results", [])) else {}
                _store_embedding(session_id, message, ticket, ticket_result, platform)
        else:
            _store_embedding(session_id, message, details, task_result, platform)
    except Exception as e:
        # Embedding failure is non-fatal — never block the main response
        print(f"Embedding storage error (non-fatal): {e}")

    # ---------------------------------------------------------------------------
    # Response
    # ---------------------------------------------------------------------------
    response_data = {
        "response": final_response,
        "platform": platform
    }

    if file_filename:
        response_data["file_processed"] = file_filename
        response_data["file_type"]      = file_type

    if file_error:
        response_data["file_error"] = file_error

    return jsonify(response_data)


# ---------------------------------------------------------------------------
# EMBEDDING HELPER
# ---------------------------------------------------------------------------

def _store_embedding(session_id: str, message: str, details: dict,
                     task_result: dict, platform: str):
    """
    Generate and store a vector embedding for a ticket or conversation.
    This powers the RAG retrieval in context_agent.

    The embedding captures:
    - The original user message
    - The ticket summary and type
    - The ticket ID and platform
    So future queries like "find the login bug" retrieve this record.
    """
    ticket_id  = task_result.get("ticket_id", "")
    issue_type = details.get("issue_type", "") if isinstance(details, dict) else ""
    summary    = details.get("summary", "")    if isinstance(details, dict) else ""
    labels     = details.get("labels", [])     if isinstance(details, dict) else []

    # Only store embeddings when a ticket was actually created
    if not ticket_id:
        return

    # Build rich descriptive text — more context = better semantic search
    content = (
        f"user request: {message} | "
        f"summary: {summary} | "
        f"type: {issue_type} | "
        f"labels: {', '.join(labels)} | "
        f"platform: {platform} | "
        f"ticket_id: {ticket_id}"
    )

    vector = get_embedding(content)

    db.session.add(ConversationEmbedding(
        session_id  = session_id,
        content     = content,
        ticket_id   = ticket_id,
        ticket_type = issue_type,
        platform    = platform,
        embedding   = json.dumps(vector)
    ))
    db.session.commit()
    print(f"Embedding stored for ticket {ticket_id} ({len(vector)} dims)")


# ---------------------------------------------------------------------------
# HISTORY ROUTE
# ---------------------------------------------------------------------------

@chat_bp.route("/history", methods=["GET"])
def history():
    session_id = session.get("session_id")
    if not session_id:
        return jsonify({"history": []})
    rows = Conversation.query.filter_by(
        session_id=session_id
    ).order_by(Conversation.created_at).all()
    return jsonify({"history": [r.to_dict() for r in rows]})