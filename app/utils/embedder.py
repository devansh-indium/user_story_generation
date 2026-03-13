import requests
import json
from app.config import Config

def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for a piece of text
    using Azure OpenAI text-embedding-ada-002.
    """
    endpoint   = Config.AZURE_OPENAI_VISION_ENDPOINT.rstrip("/")
    api_key    = Config.AZURE_OPENAI_API_KEY
    deployment = "text-embedding-ada-002"  # deploy this in AI Foundry

    url = (
        f"{endpoint}/openai/deployments/{deployment}"
        f"/embeddings?api-version=2024-02-15-preview"
    )

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json", "api-key": api_key},
        json={"input": text[:8000]}  # ada-002 has 8k token limit
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    dot   = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0