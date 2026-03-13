import base64
import requests
import os
from app.config import Config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif"
}


def is_image(file) -> bool:
    """Check if uploaded file is an image based on extension."""
    ext = os.path.splitext(file.filename.lower())[1]
    return ext in IMAGE_EXTENSIONS


def analyse_image(file, context: str = "") -> str:
    """
    Send an image to GPT-4o on Azure and get back a fully structured,
    exhaustive text description ready to be used in a ticket.
    """
    ext       = os.path.splitext(file.filename.lower())[1]
    mime_type = MIME_MAP.get(ext, "image/png")

    image_data   = file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    print(f"Analysing image: {file.filename} ({len(image_data)} bytes, {mime_type})")

    system_prompt = """You are a senior technical business analyst embedded in a software development team.
Your job is to analyse images and produce exhaustive, structured descriptions that developers,
designers, and QA engineers can use directly to build or test features — without ever needing
to see the original image.

Your analysis must be so thorough and precise that:
- A frontend developer can build the exact UI from your description alone
- A QA engineer can write test cases from your description alone
- A product manager can understand the full scope from your description alone

---

FOR UI SCREENSHOTS OR DESIGNS, produce a description with ALL of these sections:

## Page Overview
Describe the overall purpose and layout of the page. What is it for? Who uses it?

## Layout & Structure
Describe the exact layout. How many columns? What is in the header, sidebar, main content, footer?
What is the visual hierarchy? How is space divided?

## Navigation & Header
Describe every navigation element, logo, breadcrumb, tab, or top bar visible.
Include exact text labels on all nav items.

## Main Content Area
Describe every section in detail:
- Exact headings and subheadings visible
- All cards, tiles, or list items — include their exact text, icons, tags, badges
- All images, illustrations, or icons and what they represent
- Tables — include column headers and sample data visible
- Any charts or graphs — type, axes, data shown

## UI Components
List every interactive component:
- Buttons — exact label text, position, color, style (primary/secondary/outline)
- Input fields — labels, placeholder text, type (text/search/dropdown)
- Dropdowns or selects — options visible
- Checkboxes, radio buttons, toggles — labels and state (checked/unchecked)
- Modals, tooltips, popovers — content if visible
- Tabs — all tab labels and which is active

## Typography & Styling
- Font sizes (approximate: large heading, medium subheading, body, small label)
- Color scheme — primary colors, background colors, accent colors
- Any dark/light mode indicators

## State & Data
- What data is currently displayed? (e.g. list of items, empty state, loading state)
- Are there any visible counts, numbers, percentages, or metrics?
- Is there pagination? How many items per page?

## User Interactions & Flows
- What can the user do on this page?
- What happens when they click each interactive element?
- Are there any visible hover states, active states, or animations implied?

## Error States & Edge Cases
- Are there any error messages, warnings, or validation messages visible?
- Is there an empty state shown?

## Accessibility & Responsiveness Clues
- Any visible indicators of responsive design?
- Any accessibility features visible (aria labels, focus states, etc.)?

---

FOR BUG SCREENSHOTS, produce:

## What Is Broken
Describe exactly what error, broken UI, or unexpected state is visible.

## Error Details
Copy any error messages, codes, or stack traces visible verbatim.

## UI State at Time of Error
What was the user doing? What does the page look like? What is broken vs working?

## Environment Clues
Browser, OS, device, URL, any version numbers visible in the screenshot.

## Impact Assessment
How severe does this appear? What functionality is blocked?

---

Be exhaustive. Do not summarize. Do not skip sections.
If a section is not applicable write "Not visible in screenshot."
"""

    user_text = (
        "Analyse this image exhaustively using all the sections in your instructions. "
        "Do not skip any section. Be as detailed as possible — include exact text, "
        "exact labels, exact colors, exact component names as you see them.\n\n"
        + (f"Additional context from user: {context}" if context else "")
    )

    payload = {
        "messages": [
            {
                "role":    "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    endpoint   = Config.AZURE_OPENAI_VISION_ENDPOINT.rstrip("/")
    deployment = Config.AZURE_OPENAI_VISION_DEPLOYMENT
    api_key    = Config.AZURE_OPENAI_API_KEY

    url = (
        f"{endpoint}/openai/deployments/{deployment}"
        f"/chat/completions?api-version=2025-01-01-preview"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key":      api_key
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(
            f"GPT-4o vision API error {resp.status_code}: {resp.text}"
        )

    result      = resp.json()
    description = result["choices"][0]["message"]["content"]
    print(f"Image analysis complete: {len(description)} chars")

    return description