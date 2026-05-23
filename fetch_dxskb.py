"""Fetch Commerce KB articles from DXSKB Jira project."""
import json
import time
import requests
from config import JIRA_BASE, JIRA_PAT, DATA_DIR

HEADERS = {"Authorization": f"Bearer {JIRA_PAT}"}
OUTPUT = DATA_DIR / "dxskb_articles.json"

JQL = (
    'project = DXSKB '
    'AND issuetype = Documentation '
    'AND labels in ("commerce", "commerce-tier3") '
    'AND status in (Done, Approved) '
    'ORDER BY created DESC'
)
FIELDS = "key,summary,description,labels,status,created"


def clean_description(text: str) -> str:
    """Strip Jira code block markup from article content."""
    for tag in ["{code:java}", "{code:sql}", "{code:bash}", "{code:php}", "{code}"]:
        text = text.replace(tag, "")
    return text.strip()


def fetch():
    start, page_size = 0, 100
    articles = []

    while True:
        r = requests.get(
            f"{JIRA_BASE}/rest/api/2/search",
            headers=HEADERS,
            params={
                "jql": JQL,
                "fields": FIELDS,
                "maxResults": page_size,
                "startAt": start,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        total = data["total"]
        issues = data.get("issues", [])

        if not issues:
            break

        for issue in issues:
            flds = issue["fields"]
            labels = flds.get("labels") or []
            desc = clean_description(flds.get("description") or "")
            articles.append({
                "key": issue["key"],
                "summary": flds.get("summary") or "",
                "description": desc[:2000],
                "labels": labels,
                "status": flds["status"]["name"],
                "is_public": "public" in labels,
                "created": flds.get("created") or "",
            })

        print(f"  Fetched {len(articles)}/{total}")
        start += page_size
        if start >= total:
            break
        time.sleep(0.1)

    with open(OUTPUT, "w") as f:
        json.dump(articles, f, indent=2)

    public = sum(1 for a in articles if a["is_public"])
    internal = len(articles) - public
    print(f"\nDone. Saved {len(articles)} DXSKB Commerce articles → {OUTPUT}")
    print(f"  Publicly documented : {public}")
    print(f"  Internally documented (no public label): {internal}")


if __name__ == "__main__":
    fetch()
