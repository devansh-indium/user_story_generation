import requests
from requests.auth import HTTPBasicAuth
from app.config import Config


class JiraMCP:
    """
    Jira Cloud REST API v3 client.
    Handles all direct communication with Jira.
    """

    def __init__(self):
        self.base_url   = Config.JIRA_BASE_URL
        self.auth       = HTTPBasicAuth(Config.JIRA_EMAIL, Config.JIRA_API_TOKEN)
        self.headers    = {"Accept": "application/json", "Content-Type": "application/json"}
        self.project_key = Config.JIRA_PROJECT_KEY

   
    def _url(self, path: str) -> str:
        """Build full Jira API URL."""
        return f"{self.base_url}/rest/api/3/{path}"

    def _adf_doc(self, text: str) -> dict:
        """
        Wrap plain text into Atlassian Document Format (ADF).
        Jira Cloud requires description in ADF, not plain strings.
        """
        return {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": text}]
                }
            ]
        }

    def _adf_full(self, description: str, user_story: str,
                  acceptance_criteria: list, gherkin: str) -> dict:
        """
        Build a rich ADF document with structured sections:
        - User Story
        - Description
        - Acceptance Criteria (bullet list)
        - Gherkin scenarios (code block)
        """
        content = []

        # --- User Story section ---
        if user_story:
            content += [
                {"type": "heading", "attrs": {"level": 3},
                 "content": [{"type": "text", "text": "User Story"}]},
                {"type": "paragraph",
                 "content": [{"type": "text", "text": user_story,
                               "marks": [{"type": "em"}]}]}
            ]

        # --- Description section ---
        if description:
            content += [
                {"type": "heading", "attrs": {"level": 3},
                 "content": [{"type": "text", "text": "Description"}]},
                {"type": "paragraph",
                 "content": [{"type": "text", "text": description}]}
            ]

        # --- Acceptance Criteria section ---
        if acceptance_criteria:
            items = [
                {"type": "listItem", "content": [
                    {"type": "paragraph",
                     "content": [{"type": "text", "text": criterion}]}
                ]}
                for criterion in acceptance_criteria
            ]
            content += [
                {"type": "heading", "attrs": {"level": 3},
                 "content": [{"type": "text", "text": "Acceptance Criteria"}]},
                {"type": "bulletList", "content": items}
            ]

        # --- Gherkin section ---
        if gherkin:
            content += [
                {"type": "heading", "attrs": {"level": 3},
                 "content": [{"type": "text", "text": "Gherkin Scenarios"}]},
                {"type": "codeBlock", "attrs": {"language": "gherkin"},
                 "content": [{"type": "text", "text": gherkin}]}
            ]

        return {"version": 1, "type": "doc", "content": content}

    

    def create_issue(self, details: dict) -> dict:
        """
        Create a new Jira issue.
        Supports both simple and complex (spec + Gherkin) tickets.
        """
        project_key      = details.get("project") or self.project_key
        summary          = details.get("summary", "New Ticket")
        description      = details.get("description", "")
        priority         = details.get("priority", "Medium")
        issue_type       = details.get("issue_type", "Story")
        labels           = details.get("labels", [])
        story_points     = details.get("story_points")
        user_story       = details.get("user_story", "")
        acceptance_criteria = details.get("acceptance_criteria", [])
        gherkin          = details.get("gherkin", "")

        # Use rich ADF if we have spec data, otherwise simple paragraph
        if user_story or acceptance_criteria or gherkin:
            adf_description = self._adf_full(
                description, user_story, acceptance_criteria, gherkin
            )
        else:
            adf_description = self._adf_doc(description) if description else self._adf_doc("")

        payload = {
            "fields": {
                "project":     {"key": project_key},
                "summary":     summary,
                "description": adf_description,
                "issuetype":   {"name": issue_type},
                "priority":    {"name": priority},
                "labels":      labels if isinstance(labels, list) else []
            }
        }

        
        if story_points:
            payload["fields"]["customfield_10016"] = story_points

        resp = requests.post(
            self._url("issue"),
            json=payload,
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 201:
            data = resp.json()
            return {
                "status": "success",
                "ticket_id": data.get("key"),
                "ticket_url": f"{self.base_url}/browse/{data.get('key')}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    # -------------------------------------------------------------------------
    # UPDATE ISSUE
    # -------------------------------------------------------------------------

    def update_issue(self, details: dict) -> dict:
        """Update fields on an existing Jira issue."""
        ticket_id = details.get("ticket_id")
        if not ticket_id:
            return {"status": "error", "detail": "ticket_id is required"}

        fields = {}
        if details.get("summary"):
            fields["summary"] = details["summary"]
        if details.get("description"):
            fields["description"] = self._adf_doc(details["description"])
        if details.get("priority"):
            fields["priority"] = {"name": details["priority"]}

        if not fields:
            return {"status": "error", "detail": "No fields to update provided"}

        resp = requests.put(
            self._url(f"issue/{ticket_id}"),
            json={"fields": fields},
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 204:
            return {
                "status": "success",
                "ticket_id": ticket_id,
                "ticket_url": f"{self.base_url}/browse/{ticket_id}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    
    def get_issue(self, ticket_id: str) -> dict:
        """Get full details of a Jira issue."""
        resp = requests.get(
            self._url(f"issue/{ticket_id}"),
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 200:
            data   = resp.json()
            fields = data.get("fields", {})
            return {
                "status":      "success",
                "ticket_id":   data.get("key"),
                "summary":     fields.get("summary", ""),
                "status":      fields.get("status", {}).get("name", ""),
                "priority":    fields.get("priority", {}).get("name", ""),
                "assignee":    (fields.get("assignee") or {}).get("displayName", "Unassigned"),
                "issue_type":  fields.get("issuetype", {}).get("name", ""),
                "labels":      fields.get("labels", []),
                "ticket_url":  f"{self.base_url}/browse/{data.get('key')}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    # -------------------------------------------------------------------------
    # SEARCH ISSUES
    # -------------------------------------------------------------------------

    def search_issues(self, details: dict) -> dict:
        """Search Jira issues using JQL."""
        project  = details.get("project") or self.project_key
        status   = details.get("status", "")
        assignee = details.get("assignee", "")
        keyword  = details.get("keyword", "")

        jql_parts = [f"project = {project}"]
        if status:
            jql_parts.append(f'status = "{status}"')
        if assignee:
            jql_parts.append(f'assignee = "{assignee}"')
        if keyword:
            jql_parts.append(f'text ~ "{keyword}"')

        jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

        resp = requests.get(
            self._url("search/jql"),
            params={"jql": jql, "maxResults": 20},
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 200:
            data   = resp.json()
            issues = data.get("issues", [])
            return {
                "status": "success",
                "total": len(issues),
                "issues": [
                    {
                        "ticket_id":  i.get("key"),
                        "summary":    i.get("fields", {}).get("summary", ""),
                        "status":     i.get("fields", {}).get("status", {}).get("name", ""),
                        "priority":   i.get("fields", {}).get("priority", {}).get("name", ""),
                        "assignee":   (i.get("fields", {}).get("assignee") or {}).get("displayName", "Unassigned"),
                        "ticket_url": f"{self.base_url}/browse/{i.get('key')}"
                    }
                    for i in issues
                ]
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    
    def get_transitions(self, ticket_id: str) -> dict:
        """
        Get all available status transitions for a ticket.
        Use this before transition_issue to find valid transition names.
        """
        resp = requests.get(
            self._url(f"issue/{ticket_id}/transitions"),
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 200:
            transitions = resp.json().get("transitions", [])
            return {
                "status": "success",
                "ticket_id": ticket_id,
                "transitions": [
                    {"id": t.get("id"), "name": t.get("name")}
                    for t in transitions
                ]
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

   
    def transition_issue(self, details: dict) -> dict:
        """
        Move a Jira issue to a new status.
        Looks up the transition ID by name automatically.
        e.g. transition_name = "In Progress", "Done", "To Do"
        """
        ticket_id       = details.get("ticket_id")
        transition_name = details.get("transition_name", "")

        if not ticket_id or not transition_name:
            return {"status": "error", "detail": "ticket_id and transition_name are required"}

        # Step 1 — fetch available transitions
        transitions_result = self.get_transitions(ticket_id)
        if transitions_result.get("status") == "error":
            return transitions_result

        # Step 2 — find matching transition ID (case insensitive)
        transition_id = None
        for t in transitions_result.get("transitions", []):
            if t["name"].lower() == transition_name.lower():
                transition_id = t["id"]
                break

        if not transition_id:
            available = [t["name"] for t in transitions_result.get("transitions", [])]
            return {
                "status": "error",
                "detail": f"Transition '{transition_name}' not found.",
                "available_transitions": available
            }

        
        resp = requests.post(
            self._url(f"issue/{ticket_id}/transitions"),
            json={"transition": {"id": transition_id}},
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 204:
            return {
                "status": "success",
                "ticket_id": ticket_id,
                "new_status": transition_name,
                "ticket_url": f"{self.base_url}/browse/{ticket_id}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    
    def add_comment(self, details: dict) -> dict:
        """Add a comment to an existing Jira issue."""
        ticket_id = details.get("ticket_id")
        comment   = details.get("comment", "")

        if not ticket_id or not comment:
            return {"status": "error", "detail": "ticket_id and comment are required"}

        payload = {
            "body": {
                "version": 1,
                "type": "doc",
                "content": [
                    {"type": "paragraph",
                     "content": [{"type": "text", "text": comment}]}
                ]
            }
        }

        resp = requests.post(
            self._url(f"issue/{ticket_id}/comment"),
            json=payload,
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 201:
            data = resp.json()
            return {
                "status":    "success",
                "ticket_id": ticket_id,
                "comment_id": data.get("id"),
                "ticket_url": f"{self.base_url}/browse/{ticket_id}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

   
    def link_issues(self, details: dict) -> dict:
        """
        Create a link between two Jira issues.
        link_type options: "blocks", "is blocked by", "duplicates",
                           "is duplicated by", "relates to", "clones"
        """
        ticket_id        = details.get("ticket_id")
        linked_ticket_id = details.get("linked_ticket_id")
        link_type        = details.get("link_type", "relates to")

        if not ticket_id or not linked_ticket_id:
            return {"status": "error", "detail": "ticket_id and linked_ticket_id are required"}

        payload = {
            "type":          {"name": link_type},
            "inwardIssue":   {"key": ticket_id},
            "outwardIssue":  {"key": linked_ticket_id}
        }

        resp = requests.post(
            self._url("issueLink"),
            json=payload,
            auth=self.auth,
            headers=self.headers
        )

        if resp.status_code == 201:
            return {
                "status":          "success",
                "ticket_id":       ticket_id,
                "linked_ticket_id": linked_ticket_id,
                "link_type":       link_type
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}