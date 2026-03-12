from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from app.config import Config


def get_project_client() -> AIProjectClient:
    """
    Uses DefaultAzureCredential — works automatically after
    running 'az login' locally. On Azure deployment it uses
    Managed Identity automatically. No API keys needed.
    """
    return AIProjectClient(
        endpoint   = Config.AZURE_FOUNDRY_PROJECT_ENDPOINT,
        credential = DefaultAzureCredential()
    )