import json
from app.mcp.jira_mcp import JiraMCP

jira = JiraMCP()


def create_jira_ticket(
    project: str,
    summary: str,
    description: str,
    priority: str,
    issue_type: str,
    labels: str = "[]",
    story_points: int = 0,
    user_story: str = "",
    acceptance_criteria: str = "[]",
    gherkin: str = ""
) -> str:
    """
    Creates a new Jira ticket with full spec support.
    :param project: Jira project key e.g. SCRUM
    :param summary: One-line title of the ticket
    :param description: Plain text description of the work
    :param priority: One of Lowest, Low, Medium, High, Highest
    :param issue_type: One of Story, Bug, Task, Epic, Subtask
    :param labels: JSON string list of labels e.g. '["auth", "backend"]'
    :param story_points: Estimated effort as a number e.g. 3, 5, 8
    :param user_story: User story in "As a... I want... So that..." format
    :param acceptance_criteria: JSON string list of acceptance criteria
    :param gherkin: Full Gherkin feature + scenarios as plain text
    """
    try:
        labels_list = json.loads(labels) if isinstance(labels, str) else labels
    except (json.JSONDecodeError, TypeError):
        labels_list = []

    try:
        ac_list = json.loads(acceptance_criteria) if isinstance(acceptance_criteria, str) else acceptance_criteria
    except (json.JSONDecodeError, TypeError):
        ac_list = []

    return json.dumps(jira.create_issue({
        "project":              project,
        "summary":              summary,
        "description":          description,
        "priority":             priority,
        "issue_type":           issue_type,
        "labels":               labels_list,
        "story_points":         story_points if story_points else None,
        "user_story":           user_story,
        "acceptance_criteria":  ac_list,
        "gherkin":              gherkin
    }))


def update_jira_ticket(
    ticket_id: str,
    summary: str = "",
    description: str = "",
    priority: str = ""
) -> str:
    """
    Updates fields on an existing Jira ticket.
    :param ticket_id: Jira ticket ID e.g. SCRUM-42
    :param summary: New summary text (optional)
    :param description: New description text (optional)
    :param priority: New priority: Lowest, Low, Medium, High, Highest (optional)
    """
    details = {"ticket_id": ticket_id}
    if summary:     details["summary"]     = summary
    if description: details["description"] = description
    if priority:    details["priority"]    = priority
    return json.dumps(jira.update_issue(details))


def search_jira_tickets(
    project: str,
    status: str = "",
    assignee: str = "",
    keyword: str = ""
) -> str:
    """
    Searches Jira tickets using JQL filters.
    :param project: Jira project key e.g. SCRUM
    :param status: Filter by status e.g. "To Do", "In Progress", "Done"
    :param assignee: Filter by assignee display name or account ID
    :param keyword: Search keyword in summary/description
    """
    return json.dumps(jira.search_issues({
        "project":  project,
        "status":   status,
        "assignee": assignee,
        "keyword":  keyword
    }))


def get_jira_ticket(ticket_id: str) -> str:
    """
    Gets full details of a specific Jira ticket.
    :param ticket_id: Jira ticket ID e.g. SCRUM-42
    """
    return json.dumps(jira.get_issue(ticket_id))


def transition_jira_ticket(ticket_id: str, transition_name: str) -> str:
    """
    Moves a Jira ticket to a new status.
    :param ticket_id: Jira ticket ID e.g. SCRUM-42
    :param transition_name: Target status e.g. "In Progress", "Done", "To Do"
    """
    return json.dumps(jira.transition_issue({
        "ticket_id":       ticket_id,
        "transition_name": transition_name
    }))


def add_comment_to_ticket(ticket_id: str, comment: str) -> str:
    """
    Adds a comment to an existing Jira ticket.
    :param ticket_id: Jira ticket ID e.g. SCRUM-42
    :param comment: The comment text to add
    """
    return json.dumps(jira.add_comment({
        "ticket_id": ticket_id,
        "comment":   comment
    }))


def link_jira_tickets(
    ticket_id: str,
    linked_ticket_id: str,
    link_type: str = "relates to"
) -> str:
    """
    Creates a link between two Jira tickets.
    :param ticket_id: Source ticket ID e.g. SCRUM-42
    :param linked_ticket_id: Target ticket ID e.g. SCRUM-43
    :param link_type: Relationship type e.g. "blocks", "is blocked by", "duplicates", "relates to"
    """
    return json.dumps(jira.link_issues({
        "ticket_id":        ticket_id,
        "linked_ticket_id": linked_ticket_id,
        "link_type":        link_type
    }))


def get_jira_transitions(ticket_id: str) -> str:
    """
    Gets all available status transitions for a Jira ticket.
    Use this to find valid transition names before calling transition_jira_ticket.
    :param ticket_id: Jira ticket ID e.g. SCRUM-42
    """
    return json.dumps(jira.get_transitions(ticket_id))



jira_functions = {
    create_jira_ticket,
    update_jira_ticket,
    search_jira_tickets,
    get_jira_ticket,
    transition_jira_ticket,
    add_comment_to_ticket,
    link_jira_tickets,
    get_jira_transitions
}