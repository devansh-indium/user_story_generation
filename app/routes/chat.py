from flask import Blueprint, request, jsonify, session
from app import db
from app.models.conversation import Conversation
from app.agents.input_agent import process_input
from app.agents.context_agent import enrich_with_context
from app.agents.jira_agent import execute_jira_task
from app.agents.response_agent import generate_response
from app.utils.file_reader import extract_text, truncate_text
import uuid

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    # ---------------------------------------------------------------------------
    # Parse request — supports both JSON and multipart/form-data (file uploads)
    # ---------------------------------------------------------------------------
    if request.content_type and "multipart/form-data" in request.content_type:
        # File upload request
        message  = request.form.get("message", "")
        platform = request.form.get("platform", "jira")
        file     = request.files.get("file")
    else:
        # Regular JSON request
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
    # Extract file text if a file was uploaded
    # ---------------------------------------------------------------------------
    file_text     = None
    file_error    = None
    file_filename = None

    if file and file.filename:
        try:
            raw_text      = extract_text(file)
            file_text     = truncate_text(raw_text, max_chars=8000)
            file_filename = file.filename
            print(f"File uploaded: {file_filename} ({len(file_text)} chars extracted)")
        except (ValueError, ImportError) as e:
            file_error = str(e)
            print(f"File read error: {file_error}")

    # ---------------------------------------------------------------------------
    # Build the full message for the agent
    # ---------------------------------------------------------------------------
    if file_text:
        full_message = (
            f"{message}\n\n"
            f"--- ATTACHED DOCUMENT: {file_filename} ---\n"
            f"{file_text}\n"
            f"--- END OF DOCUMENT ---"
        )
    else:
        full_message = message

    # Log for debugging
    print(f"Received message: {message} | Platform: {platform}")
    if file_filename:
        print(f"With attached file: {file_filename}")

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

    enriched = enrich_with_context(history, input_result)
    print(f"Enriched request: {enriched}")

    task_result = execute_jira_task(enriched, platform=platform)
    print(f"Task execution result: {task_result}")

    final_response = generate_response(task_result, message)
    print(f"Final response: {final_response}")

    # ---------------------------------------------------------------------------
    # Save to database
    # ---------------------------------------------------------------------------
    user_content = message
    if file_filename:
        user_content += f" [Attached: {file_filename}]"

    db.session.add(Conversation(session_id=session_id, role="user",      content=user_content))
    db.session.add(Conversation(session_id=session_id, role="assistant", content=final_response))
    db.session.commit()

    # ---------------------------------------------------------------------------
    # Response
    # ---------------------------------------------------------------------------
    response_data = {
        "response": final_response,
        "platform": platform
    }

    if file_filename:
        response_data["file_processed"] = file_filename

    if file_error:
        response_data["file_error"] = file_error

    return jsonify(response_data)


@chat_bp.route("/history", methods=["GET"])
def history():
    session_id = session.get("session_id")
    if not session_id:
        return jsonify({"history": []})
    rows = Conversation.query.filter_by(
        session_id=session_id
    ).order_by(Conversation.created_at).all()
    return jsonify({"history": [r.to_dict() for r in rows]})