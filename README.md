# Jira Multi-Agent System — Setup Guide

---

## 1. Install System Dependencies

### Mac
- Install Homebrew: https://brew.sh
- `brew install python@3.11`
- `brew install unixodbc`
- `brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release`
- `HOMEBREW_ACCEPT_EULA=Y brew install msodbcsql18`
- `brew install azure-cli`

### Windows
- Install Python 3.11 (check **Add to PATH**): https://www.python.org/downloads/
- Install ODBC Driver 18 for SQL Server (x64): https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
- Install Azure CLI: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows

---

## 2. Clone the Repository

```bash
git clone <your-repo-url>
cd jira_multiagent
```

---

## 3. Set Up Azure SQL Database

1. Go to https://portal.azure.com
2. Create a **SQL Server** — note the server name, admin username and password
3. Create a **SQL Database** under that server — Basic tier is fine
4. In the server **Networking** settings add your IP to the firewall and enable **Allow Azure services**
5. Your connection string for `.env` will be:
```
mssql+pyodbc://<user>:<password>@<server>.database.windows.net/<dbname>?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
```

---

## 4. Set Up Azure AI Foundry

1. Go to https://ai.azure.com
2. Create a new **Hub** then a **Project** under it
3. Copy the **Project endpoint** from Project Settings — you will need it in `.env`

---

## 5. Deploy the Model

1. Inside your Foundry project go to **Models + endpoints** → **Deploy base model**
2. Search for and deploy `gpt-oos-120b` with deployment name `gpt-oos-120b`
3. From the project **Overview** copy the **Azure OpenAI endpoint** and **API Key**

---

## 6. Get Jira Credentials

1. Your Jira base URL is `https://yourcompany.atlassian.net`
2. Generate an API token at: https://id.atlassian.com/manage-profile/security/api-tokens
3. Your project key is the prefix on your ticket numbers e.g. `SCRUM`

---

## 7. Configure Environment Variables

Create a `.env` file in the project root and fill in all values:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_FOUNDRY_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
DATABASE_URL=mssql+pyodbc://user:password@server.database.windows.net/dbname?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
FLASK_SECRET_KEY=any-random-string
FLASK_DEBUG=True
JIRA_BASE_URL=https://yourcompany.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=SCRUM
```

---

## 8. Create Virtual Environment and Install Packages

### Mac
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Every time you open a new terminal, activate the venv before running the app.

---

## 9. Run Database Migrations

```bash
flask db upgrade
```

---

## 10. Run the App

Log in to Azure first:
```bash
az login
```

Then start the server:
```bash
python3.11 run.py
```

App will be running at `http://127.0.0.1:5000`

---

## 11. Testing with Postman

- **Method**: POST
- **URL**: `http://127.0.0.1:5000/chat`
- **Body**: raw → JSON

```json
{
    "message": "create a bug ticket for login crash on mobile, high priority"
}
```

To check conversation history: `GET http://127.0.0.1:5000/history`

---
