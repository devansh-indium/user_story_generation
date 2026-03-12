import json
from app.mcp.devops_mcp import DevOpsMCP

devops = DevOpsMCP()


def create_devops_work_item(
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
    Creates a new Azure DevOps work item with full spec support.
    :param summary: One-line title of the work item
    :param description: Plain text description of the work
    :param priority: One of Lowest, Low, Medium, High, Highest
    :param issue_type: One of User Story, Bug, Task, Epic, Feature
    :param labels: JSON string list of tags e.g. '["auth", "backend"]'
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

    return json.dumps(devops.create_work_item({
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


def update_devops_work_item(
    ticket_id: str,
    summary: str = "",
    description: str = "",
    priority: str = ""
) -> str:
    """
    Updates fields on an existing Azure DevOps work item.
    :param ticket_id: Work item ID e.g. 42
    :param summary: New title text (optional)
    :param description: New description text (optional)
    :param priority: New priority: Lowest, Low, Medium, High, Highest (optional)
    """
    details = {"ticket_id": ticket_id}
    if summary:     details["summary"]     = summary
    if description: details["description"] = description
    if priority:    details["priority"]    = priority
    return json.dumps(devops.update_work_item(details))


def search_devops_work_items(
    status: str = "",
    assignee: str = "",
    keyword: str = ""
) -> str:
    """
    Searches Azure DevOps work items using WIQL filters.
    :param status: Filter by state e.g. "New", "Active", "Resolved", "Closed"
    :param assignee: Filter by assignee display name
    :param keyword: Search keyword in title
    """
    return json.dumps(devops.search_work_items({
        "status":   status,
        "assignee": assignee,
        "keyword":  keyword
    }))


def get_devops_work_item(ticket_id: str) -> str:
    """
    Gets full details of a specific Azure DevOps work item.
    :param ticket_id: Work item ID e.g. 42
    """
    return json.dumps(devops.get_work_item(ticket_id))


def transition_devops_work_item(ticket_id: str, transition_name: str) -> str:
    """
    Moves an Azure DevOps work item to a new state.
    :param ticket_id: Work item ID e.g. 42
    :param transition_name: Target state e.g. "Active", "Resolved", "Closed"
    """
    return json.dumps(devops.transition_work_item({
        "ticket_id":       ticket_id,
        "transition_name": transition_name
    }))


def add_comment_to_work_item(ticket_id: str, comment: str) -> str:
    """
    Adds a comment to an existing Azure DevOps work item.
    :param ticket_id: Work item ID e.g. 42
    :param comment: The comment text to add
    """
    return json.dumps(devops.add_comment({
        "ticket_id": ticket_id,
        "comment":   comment
    }))


def link_devops_work_items(
    ticket_id: str,
    linked_ticket_id: str,
    link_type: str = "relates to"
) -> str:
    """
    Creates a link between two Azure DevOps work items.
    :param ticket_id: Source work item ID e.g. 42
    :param linked_ticket_id: Target work item ID e.g. 43
    :param link_type: Relationship type e.g. "relates to", "blocks", "is blocked by", "parent", "child", "duplicates"
    """
    return json.dumps(devops.link_work_items({
        "ticket_id":        ticket_id,
        "linked_ticket_id": linked_ticket_id,
        "link_type":        link_type
    }))


def get_devops_states(work_item_type: str = "User Story") -> str:
    """
    Gets all available states for a work item type.
    Use this before transitioning to find valid state names.
    :param work_item_type: e.g. "User Story", "Bug", "Task", "Epic"
    """
    return json.dumps(devops.get_states(work_item_type))


# All functions registered as tools for the jira_agent FunctionTool
devops_functions = {
    create_devops_work_item,
    update_devops_work_item,
    search_devops_work_items,
    get_devops_work_item,
    transition_devops_work_item,
    add_comment_to_work_item,
    link_devops_work_items,
    get_devops_states
}