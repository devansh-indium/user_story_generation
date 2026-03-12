import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ─── Flask ───────────────────────────────────────────
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"

    # ─── Database ────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ─── Azure OpenAI ────────────────────────────────────
    # 👇 Change model here in ONE place for all agents
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-oss-120b")
    AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_FOUNDRY_PROJECT_ENDPOINT = os.getenv("AZURE_FOUNDRY_PROJECT_ENDPOINT")
    # ─── Jira ─────────────────────────────────────────────
    JIRA_BASE_URL     = os.getenv("JIRA_BASE_URL")
    JIRA_EMAIL        = os.getenv("JIRA_EMAIL")
    JIRA_API_TOKEN    = os.getenv("JIRA_API_TOKEN")
    JIRA_PROJECT_KEY  = os.getenv("JIRA_PROJECT_KEY", "SCRUM")

    AZURE_DEVOPS_ORG     = os.getenv("AZURE_DEVOPS_ORG")
    AZURE_DEVOPS_PROJECT = os.getenv("AZURE_DEVOPS_PROJECT")
    AZURE_DEVOPS_PAT     = os.getenv("AZURE_DEVOPS_PAT")
