"""Fetch resolved Commerce T3 (ACSD) tickets from Jira."""
import json
import time
import requests
from tqdm import tqdm
from config import JIRA_BASE, JIRA_PAT, DATA_DIR

HEADERS = {"Authorization": f"Bearer {JIRA_PAT}"}
OUTPUT = DATA_DIR / "acsd_tickets.jsonl"

JQL = (
    'project = ACSD '
    'AND status in (Resolved, Done, Closed) '
    'AND created >= -540d '
    'AND description is not EMPTY '
    'AND summary !~ "Clone of" '
    'AND summary !~ "Backport" '
    'AND summary !~ "[Backport]" '
    'AND issuetype != Sub-task '
    'ORDER BY created DESC'
)
FIELDS = "key,summary,description,comment,labels,created,resolutiondate,priority"


def best_resolution(comments: list) -> str:
    """Return the last substantive engineer comment as the resolution."""
    for c in reversed(comments):
        body = (c.get("body") or "").strip()
        if len(body) > 100:
            return body[:1000]
    return ""


def fetch():
    start, page_size = 0, 100
    total = None
    saved = 0
    pbar = None

    with open(OUTPUT, "w") as f:
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

            if total is None:
                total = data["total"]
                pbar = tqdm(total=total, desc="Fetching ACSD tickets")

            issues = data.get("issues", [])
            if not issues:
                break

            for issue in issues:
                flds = issue["fields"]
                comments = flds.get("comment", {}).get("comments", [])
                record = {
                    "key": issue["key"],
                    "summary": flds.get("summary") or "",
                    "description": (flds.get("description") or "")[:2000],
                    "resolution": best_resolution(comments),
                    "labels": flds.get("labels") or [],
                    "created": flds.get("created") or "",
                    "priority": (flds.get("priority") or {}).get("name") or "",
                }
                f.write(json.dumps(record) + "\n")
                saved += 1
                pbar.update(1)

            start += page_size
            if start >= total:
                break
            time.sleep(0.1)

    if pbar:
        pbar.close()
    print(f"\nDone. Saved {saved} tickets → {OUTPUT}")


if __name__ == "__main__":
    fetch()
