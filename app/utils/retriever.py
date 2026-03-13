import json
from app.utils.embedder import get_embedding, cosine_similarity
from app.models.conversation import ConversationEmbedding

def retrieve_relevant_context(query: str, session_id: str, top_k: int = 5) -> list[dict]:
    """
    Given a user query, find the most semantically similar
    past conversations and tickets stored in Azure SQL.
    """
    query_vector = get_embedding(query)

    # Pull all embeddings for this session from Azure SQL
    rows = ConversationEmbedding.query.filter_by(session_id=session_id).all()

    if not rows:
        return []

    # Score each row by cosine similarity
    scored = []
    for row in rows:
        stored_vector = json.loads(row.embedding)
        score = cosine_similarity(query_vector, stored_vector)
        scored.append({
            "score":       score,
            "content":     row.content,
            "ticket_id":   row.ticket_id,
            "ticket_type": row.ticket_type,
            "platform":    row.platform,
            "created_at":  str(row.created_at)
        })

    # Sort by similarity score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Return top K above a relevance threshold
    return [r for r in scored[:top_k] if r["score"] > 0.75]