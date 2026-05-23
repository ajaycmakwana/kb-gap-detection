"""
Fetch Adobe Commerce KB articles from public AdobeDocs GitHub repos.

Repos indexed:
  1. AdobeDocs/commerce-knowledge-base.en  — support KB (~227 articles)
  2. AdobeDocs/commerce-operations.en      — config/install/upgrade/perf guides
  3. AdobeDocs/commerce-on-cloud.en        — cloud infrastructure guide

All articles are publicly available on Experience League.
No auth needed (public repos). Saved to data/kb_github_articles.json.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from config import DATA_DIR

OUTPUT = DATA_DIR / "kb_github_articles.json"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TITLE_RE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
DESC_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,4}\s+", re.MULTILINE)
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")

# Repos to index.
# subdirs=None means all of help/; otherwise only the listed subdirs are walked.
# dir_map remaps git directory names to their EL URL path segments (from TOC.md anchors).
REPOS = [
    {
        "url": "https://github.com/AdobeDocs/commerce-knowledge-base.en.git",
        "el_base": "https://experienceleague.adobe.com/en/docs/commerce-knowledge-base/kb",
        "source": "commerce-knowledge-base.en",
        "subdirs": None,
        "dir_map": {},
    },
    {
        "url": "https://github.com/AdobeDocs/commerce-operations.en.git",
        "el_base": "https://experienceleague.adobe.com/en/docs/commerce-operations",
        "source": "commerce-operations.en",
        "subdirs": [
            "configuration",
            "installation",
            "performance",
            "security-and-compliance",
            "tools",
            "upgrade",
        ],
        "dir_map": {
            "configuration": "configuration-guide",
            "installation": "installation-guide",
            "performance": "performance-best-practices",
            "upgrade": "upgrade-guide",
        },
    },
    {
        "url": "https://github.com/AdobeDocs/commerce-on-cloud.en.git",
        "el_base": "https://experienceleague.adobe.com/en/docs/commerce-on-cloud",
        "source": "commerce-on-cloud.en",
        "subdirs": ["cloud-guide", "get-started"],
        "dir_map": {
            "cloud-guide": "user-guide",
            "get-started": "start",
        },
    },
]


def slug_to_el_url(rel_path: str, el_base: str, dir_map: dict) -> str:
    """Convert help/subdir/foo/bar.md → EL URL, applying dir_map for top-level segment."""
    slug = rel_path.removeprefix("help/").removesuffix(".md")
    parts = slug.split("/")
    if parts[0] in dir_map:
        parts[0] = dir_map[parts[0]]
    return f"{el_base}/{'/'.join(parts)}"


def clean_body(text: str) -> str:
    text = CODE_BLOCK_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = HEADING_RE.sub(" ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_article(
    path: Path,
    repo_root: Path,
    el_base: str,
    dir_map: dict,
    source: str,
) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    fm_match = FRONTMATTER_RE.match(raw)
    frontmatter = fm_match.group(1) if fm_match else ""
    body_start = fm_match.end() if fm_match else 0
    body = raw[body_start:]

    title_m = TITLE_RE.search(frontmatter)
    desc_m = DESC_RE.search(frontmatter)

    title = title_m.group(1).strip() if title_m else ""
    description = desc_m.group(1).strip() if desc_m else ""

    if not title:
        h1 = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = h1.group(1).strip() if h1 else path.stem.replace("-", " ").title()

    if path.stem.lower() in ("overview", "index", "toc"):
        return None

    rel = path.relative_to(repo_root / "help").as_posix()
    # Use relative path (slashes → dashes) as a unique key across repos
    key = rel.removesuffix(".md").replace("/", "--")
    el_url = slug_to_el_url(rel, el_base, dir_map)
    body_text = clean_body(body)

    return {
        "key": key,
        "summary": title,
        "description": description,
        "body_excerpt": body_text[:1500],
        "full_text": f"{title}. {description}. {body_text}"[:2000],
        "el_url": el_url,
        "source": source,
        "is_public": True,
    }


def fetch_repo(repo_cfg: dict, tmpdir: str) -> list:
    source = repo_cfg["source"]
    subdirs = repo_cfg["subdirs"]

    print(f"\nCloning {source} (shallow)...")
    repo_path = Path(tmpdir) / source
    subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", repo_cfg["url"], str(repo_path)],
        check=True,
    )

    help_dir = repo_path / "help"
    if subdirs is None:
        scan_dirs = [help_dir]
    else:
        scan_dirs = [help_dir / d for d in subdirs if (help_dir / d).exists()]

    md_files = [
        p for d in scan_dirs for p in d.rglob("*.md")
        if p.stem.lower() not in ("toc", "overview", "index")
    ]
    print(f"  Found {len(md_files)} markdown files")

    articles = []
    for p in sorted(md_files):
        article = parse_article(
            p, repo_path,
            repo_cfg["el_base"], repo_cfg["dir_map"], source,
        )
        if article:
            articles.append(article)

    print(f"  Parsed {len(articles)} articles")
    return articles


def fetch():
    all_articles = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for repo_cfg in REPOS:
            all_articles.extend(fetch_repo(repo_cfg, tmpdir))

    with open(OUTPUT, "w") as f:
        json.dump(all_articles, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total articles saved : {len(all_articles)} → {OUTPUT}")
    for repo_cfg in REPOS:
        src = repo_cfg["source"]
        count = sum(1 for a in all_articles if a["source"] == src)
        print(f"  {src}: {count}")


if __name__ == "__main__":
    fetch()
