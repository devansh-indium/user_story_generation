import json
import base64
import requests
from app.config import Config


class DevOpsMCP:
    """
    Azure DevOps REST API client.
    Handles all direct communication with Azure DevOps Boards.
    """

    def __init__(self):
        self.org         = Config.AZURE_DEVOPS_ORG
        self.project     = Config.AZURE_DEVOPS_PROJECT
        self.pat         = Config.AZURE_DEVOPS_PAT
        self.api_version = "api-version=7.1"

        # Encode PAT as base64 for Basic auth
        token = base64.b64encode(f":{self.pat}".encode()).decode()

        self.headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Basic {token}"
        }
        self.patch_headers = {
            "Content-Type":  "application/json-patch+json",
            "Authorization": f"Basic {token}"
        }

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build full Azure DevOps API URL."""
        return f"https://dev.azure.com/{self.org}/{self.project}/_apis/{path}"

    def _org_url(self, path: str) -> str:
        """Build org-level Azure DevOps API URL."""
        return f"https://dev.azure.com/{self.org}/_apis/{path}"

    def _work_item_url(self, item_id: str = "") -> str:
        """Build work items URL."""
        if item_id:
            return self._url(f"wit/workitems/{item_id}?{self.api_version}")
        return self._url(f"wit/workitems")

    def _build_patch(self, fields: dict) -> list:
        """
        Build a JSON Patch document from a dict of fields.
        Azure DevOps requires updates in JSON Patch format.
        Example: [{"op": "add", "path": "/fields/System.Title", "value": "..."}]
        """
        return [
            {"op": "add", "path": f"/fields/{key}", "value": value}
            for key, value in fields.items()
        ]

    def _markdown_to_html(self, text: str) -> str:
        """
        Convert simple markdown (## headers, - bullets, numbered lists)
        to HTML for Azure DevOps description rendering.
        """
        if not text:
            return ""
        lines  = text.split("\n")
        html   = ""
        in_ul  = False
        in_ol  = False

        for line in lines:
            # Close open lists if line is not a list item
            if not line.strip().startswith("- ") and in_ul:
                html  += "</ul>"
                in_ul  = False
            if not (line.strip() and line.strip()[0].isdigit() and ". " in line) and in_ol:
                html  += "</ol>"
                in_ol  = False

            if line.startswith("### "):
                html += f"<h4>{line[4:].strip()}</h4>"
            elif line.startswith("## "):
                html += f"<h3>{line[3:].strip()}</h3>"
            elif line.startswith("# "):
                html += f"<h2>{line[2:].strip()}</h2>"
            elif line.strip().startswith("- "):
                if not in_ul:
                    html += "<ul>"
                    in_ul = True
                html += f"<li>{line.strip()[2:]}</li>"
            elif line.strip() and line.strip()[0].isdigit() and ". " in line:
                if not in_ol:
                    html += "<ol>"
                    in_ol = True
                content = line.strip().split(". ", 1)[1] if ". " in line else line.strip()
                html += f"<li>{content}</li>"
            elif line.strip() == "":
                html += ""
            else:
                html += f"<p>{line.strip()}</p>"

        if in_ul:
            html += "</ul>"
        if in_ol:
            html += "</ol>"

        return html

    def _format_description(self, description: str, user_story: str,
                             acceptance_criteria, gherkin: str) -> str:
        """
        Build an HTML description for Azure DevOps.
        Handles industry-standard ticket format for all issue types.
        """
        html = ""

        
        if user_story:
            html += f"<blockquote><p><em>{user_story}</em></p></blockquote>"

        
        if description:
            html += self._markdown_to_html(description)

        # Acceptance criteria in Gherkin format (Story/Epic)
        if acceptance_criteria:
            ac_text = acceptance_criteria if isinstance(acceptance_criteria, str) \
                      else "\n".join(acceptance_criteria)
            if ac_text.strip():
                ac_html = ac_text[:4000].replace("\n", "<br>")
                html += f"<h3>Acceptance Criteria</h3><pre>{ac_html}</pre>"

        return html if html else f"<p>{description}</p>"

    def _map_priority(self, priority: str) -> int:
        """
        Map human-readable priority to Azure DevOps priority number.
        Azure DevOps uses 1 (Critical) to 4 (Low) instead of text.
        """
        mapping = {
            "highest":  1,
            "critical": 1,
            "high":     2,
            "medium":   3,
            "low":      4,
            "lowest":   4
        }
        return mapping.get(priority.lower(), 3)


    def _extract_repro_steps(self, description: str) -> str:
        """Extract Steps to Reproduce section and convert to HTML."""
        if "## Steps to Reproduce" in description:
            section = description.split("## Steps to Reproduce")[1]
            section = section.split("##")[0].strip()
            return self._markdown_to_html("## Steps to Reproduce\n" + section)
        return ""

    def _extract_system_info(self, description: str) -> str:
        """Extract Environment section and convert to HTML."""
        if "## Environment" in description:
            section = description.split("## Environment")[1]
            section = section.split("##")[0].strip()
            return self._markdown_to_html("## Environment\n" + section)
        return ""
    

    def create_work_item(self, details: dict) -> dict:
        """
        Create a new Azure DevOps work item.
        Supports User Story, Bug, Task, Epic, Feature.
        """
        issue_type          = details.get("issue_type", "User Story")
        summary             = details.get("summary", "New Work Item")
        description         = details.get("description", "")
        priority            = details.get("priority", "Medium")
        labels              = details.get("labels", [])
        story_points        = details.get("story_points")
        user_story          = details.get("user_story", "")
        acceptance_criteria = details.get("acceptance_criteria", [])
        gherkin             = details.get("gherkin", "")

        
        type_mapping = {
            "story":      "User Story",
            "user story": "User Story",
            "bug":        "Bug",
            "task":       "Task",
            "epic":       "Epic",
            "feature":    "Feature",
            "subtask":    "Task"
        }
        work_item_type = type_mapping.get(issue_type.lower(), issue_type)

        
        html_description = self._format_description(
            description, user_story, acceptance_criteria, gherkin
        )

        
        fields = {
            "System.Title":                   summary,
            "System.Description":             html_description,
            "Microsoft.VSTS.Common.Priority": self._map_priority(priority),
        }

        
        if work_item_type == "Bug":
            fields["Microsoft.VSTS.TCM.ReproSteps"] = self._extract_repro_steps(description)
            fields["Microsoft.VSTS.TCM.SystemInfo"]  = self._extract_system_info(description)

        if story_points:
            fields["Microsoft.VSTS.Scheduling.StoryPoints"] = story_points

        if labels:
            fields["System.Tags"] = "; ".join(labels)

        patch = self._build_patch(fields)

        resp = requests.post(
            self._url(f"wit/workitems/${work_item_type}?{self.api_version}"),
            json=patch,
            headers=self.patch_headers
        )

        if resp.status_code == 200:
            data    = resp.json()
            item_id = data.get("id")
            return {
                "status":    "success",
                "ticket_id": str(item_id),
                "ticket_url": f"https://dev.azure.com/{self.org}/{self.project}/_workitems/edit/{item_id}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}
    

    def get_work_item(self, ticket_id: str) -> dict:
        """Get full details of a specific Azure DevOps work item."""
        resp = requests.get(
            self._work_item_url(ticket_id),
            headers=self.headers
        )

        if resp.status_code == 200:
            data   = resp.json()
            fields = data.get("fields", {})
            return {
                "status":     "success",
                "ticket_id":  str(data.get("id")),
                "summary":    fields.get("System.Title", ""),
                "state":      fields.get("System.State", ""),
                "priority":   fields.get("Microsoft.VSTS.Common.Priority", ""),
                "assignee":   (fields.get("System.AssignedTo") or {}).get("displayName", "Unassigned"),
                "issue_type": fields.get("System.WorkItemType", ""),
                "labels":     fields.get("System.Tags", ""),
                "ticket_url": f"https://dev.azure.com/{self.org}/{self.project}/_workitems/edit/{data.get('id')}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    # -------------------------------------------------------------------------
    # SEARCH WORK ITEMS
    # -------------------------------------------------------------------------

    def search_work_items(self, details: dict) -> dict:
        """
        Search Azure DevOps work items using WIQL (Work Item Query Language).
        WIQL is the Azure DevOps equivalent of Jira's JQL.
        """
        status   = details.get("status", "")
        assignee = details.get("assignee", "")
        keyword  = details.get("keyword", "")

        conditions = [f"[System.TeamProject] = '{self.project}'"]

        if status:
            conditions.append(f"[System.State] = '{status}'")
        if assignee:
            conditions.append(f"[System.AssignedTo] = '{assignee}'")
        if keyword:
            conditions.append(f"[System.Title] CONTAINS '{keyword}'")

        wiql = (
            "SELECT [System.Id], [System.Title], [System.State], "
            "[System.AssignedTo], [System.WorkItemType] FROM WorkItems WHERE "
            + " AND ".join(conditions)
            + " ORDER BY [System.CreatedDate] DESC"
        )

        resp = requests.post(
            self._url(f"wit/wiql?{self.api_version}"),
            json={"query": wiql},
            headers=self.headers
        )

        if resp.status_code != 200:
            return {"status": "error", "detail": resp.text, "code": resp.status_code}

        work_item_refs = resp.json().get("workItems", [])[:20]

        if not work_item_refs:
            return {"status": "success", "total": 0, "issues": []}

        # Fetch full details for each work item
        ids          = ",".join(str(item["id"]) for item in work_item_refs)
        details_resp = requests.get(
            self._url(f"wit/workitems?ids={ids}&{self.api_version}"),
            headers=self.headers
        )

        if details_resp.status_code == 200:
            items = details_resp.json().get("value", [])
            return {
                "status": "success",
                "total":  len(items),
                "issues": [
                    {
                        "ticket_id":  str(i.get("id")),
                        "summary":    i.get("fields", {}).get("System.Title", ""),
                        "state":      i.get("fields", {}).get("System.State", ""),
                        "priority":   i.get("fields", {}).get("Microsoft.VSTS.Common.Priority", ""),
                        "assignee":   (i.get("fields", {}).get("System.AssignedTo") or {}).get("displayName", "Unassigned"),
                        "issue_type": i.get("fields", {}).get("System.WorkItemType", ""),
                        "ticket_url": f"https://dev.azure.com/{self.org}/{self.project}/_workitems/edit/{i.get('id')}"
                    }
                    for i in items
                ]
            }
        return {"status": "error", "detail": details_resp.text}

    # -------------------------------------------------------------------------
    # GET STATES  (equivalent of get_transitions in Jira)
    # -------------------------------------------------------------------------

    def get_states(self, work_item_type: str = "User Story") -> dict:
        """
        Get all available states for a work item type.
        Equivalent of get_transitions in Jira.
        """
        resp = requests.get(
            self._url(f"wit/workitemtypes/{work_item_type}/states?{self.api_version}"),
            headers=self.headers
        )

        if resp.status_code == 200:
            states = resp.json().get("value", [])
            return {
                "status": "success",
                "states": [s.get("name") for s in states]
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    # -------------------------------------------------------------------------
    # TRANSITION WORK ITEM  (update state)
    # -------------------------------------------------------------------------

    def transition_work_item(self, details: dict) -> dict:
        """
        Move a work item to a new state.
        States in DevOps Agile: New, Active, Resolved, Closed, Removed
        """
        ticket_id = details.get("ticket_id")
        new_state = details.get("transition_name", "")

        if not ticket_id or not new_state:
            return {"status": "error", "detail": "ticket_id and transition_name are required"}

        patch = self._build_patch({"System.State": new_state})

        resp = requests.patch(
            self._work_item_url(ticket_id),
            json=patch,
            headers=self.patch_headers
        )

        if resp.status_code == 200:
            return {
                "status":    "success",
                "ticket_id": ticket_id,
                "new_state": new_state,
                "ticket_url": f"https://dev.azure.com/{self.org}/{self.project}/_workitems/edit/{ticket_id}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    # -------------------------------------------------------------------------
    # ADD COMMENT
    # -------------------------------------------------------------------------

    def add_comment(self, details: dict) -> dict:
        """Add a comment to an existing Azure DevOps work item."""
        ticket_id = details.get("ticket_id")
        comment   = details.get("comment", "")

        if not ticket_id or not comment:
            return {"status": "error", "detail": "ticket_id and comment are required"}

        resp = requests.post(
            self._url(f"wit/workitems/{ticket_id}/comments?{self.api_version}"),
            json={"text": f"<p>{comment}</p>"},
            headers=self.headers
        )

        if resp.status_code == 200:
            data = resp.json()
            return {
                "status":     "success",
                "ticket_id":  ticket_id,
                "comment_id": data.get("id"),
                "ticket_url": f"https://dev.azure.com/{self.org}/{self.project}/_workitems/edit/{ticket_id}"
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}

    # -------------------------------------------------------------------------
    # LINK WORK ITEMS
    # -------------------------------------------------------------------------

    def link_work_items(self, details: dict) -> dict:
        """
        Link two Azure DevOps work items.
        link_type options: relates to, blocks, is blocked by, duplicates, parent, child
        """
        ticket_id        = details.get("ticket_id")
        linked_ticket_id = details.get("linked_ticket_id")
        link_type        = details.get("link_type", "relates to")

        if not ticket_id or not linked_ticket_id:
            return {"status": "error", "detail": "ticket_id and linked_ticket_id are required"}

        # Map simple link type names to Azure DevOps relation types
        type_mapping = {
            "relates to":    "System.LinkTypes.Related",
            "blocks":        "Microsoft.VSTS.Common.Affects-Forward",
            "is blocked by": "Microsoft.VSTS.Common.Affects-Reverse",
            "duplicates":    "System.LinkTypes.Duplicate-Forward",
            "parent":        "System.LinkTypes.Hierarchy-Reverse",
            "child":         "System.LinkTypes.Hierarchy-Forward"
        }
        relation_type = type_mapping.get(link_type.lower(), "System.LinkTypes.Related")

        patch = [{
            "op":    "add",
            "path":  "/relations/-",
            "value": {
                "rel": relation_type,
                "url": self._url(f"wit/workitems/{linked_ticket_id}"),
                "attributes": {"comment": f"Linked as {link_type}"}
            }
        }]

        resp = requests.patch(
            self._work_item_url(ticket_id),
            json=patch,
            headers=self.patch_headers
        )

        if resp.status_code == 200:
            return {
                "status":           "success",
                "ticket_id":        ticket_id,
                "linked_ticket_id": linked_ticket_id,
                "link_type":        link_type
            }
        return {"status": "error", "detail": resp.text, "code": resp.status_code}