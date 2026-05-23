"""
Embed ACSD tickets → cluster by semantic similarity → detect KB documentation gaps.
Matches clusters against DXSKB articles (internal) + public GitHub KB articles.
Results saved to data/clusters.json
"""
import json
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from config import (
    DATA_DIR, EMBED_MODEL, MIN_CLUSTER_SIZE, STRONG_THRESHOLD, WEAK_THRESHOLD
)

# Patterns to extract the meaningful sections of ACSD template descriptions.
# Tickets use structured headers like *ISSUE:*, *ACTUAL RESULTS:*, *EXPECTED RESULTS:*
# Embedding the full description buries the real problem under boilerplate template
# fields (MESH ID, QUERY URL, STEPS TO REPLICATE, etc.), causing all tickets for a
# given product area to cluster together regardless of the actual issue.
_ISSUE_RE = re.compile(
    r"\*?\b(?:ISSUE|DETAILS)\b:?\*?\s*\n(.*?)(?=\n\s*\*?\b(?:STEPS TO REPLICATE|ACTUAL RESULTS?|NOTES|ADDITIONAL|TIME AND TIMEZONE)\b)",
    re.DOTALL | re.IGNORECASE,
)
_ACTUAL_RE = re.compile(
    r"\*?\bACTUAL RESULTS?\b:?\*?\s*\n(.*?)(?=\n\s*\*?\b(?:EXPECTED|NOTES|ADDITIONAL|TIME AND TIMEZONE|DB DUMP)\b)",
    re.DOTALL | re.IGNORECASE,
)
_EXPECTED_RE = re.compile(
    r"\*?\bEXPECTED RESULTS?\b:?\*?\s*\n(.*?)(?=\n\s*\*?\b(?:NOTES|ADDITIONAL|TIME AND TIMEZONE|DB DUMP)\b)",
    re.DOTALL | re.IGNORECASE,
)
# Template header lines that pollute the embedding when no structured sections found
_TEMPLATE_NOISE_RE = re.compile(
    r"^\s*\*?(?:MESH ID|QUERY URL|REQUEST ID|API MESH (?:JSON FILE|VERSION|version)|"
    r"Communication method|Have you completed|From which Region|DB DUMP|"
    r"CODE ARCHIVE LOCATION|TIME AND TIMEZONE|Is issue reproducible|"
    r"ADOBE API MESH SPECIFIC FIELDS|ADDITIONAL INFORMATION)\b.*$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_ticket_text(ticket: dict) -> str:
    """Return problem-focused text for embedding.

    Tries to extract ISSUE + ACTUAL RESULTS + EXPECTED RESULTS sections.
    Falls back to stripping known template noise from the raw description.
    Always includes the ticket summary.
    """
    summary = ticket.get("summary", "")
    desc = ticket.get("description", "")

    parts = [summary]
    matched_any = False

    for pattern in (_ISSUE_RE, _ACTUAL_RE, _EXPECTED_RE):
        m = pattern.search(desc)
        if m:
            text = m.group(1).strip()
            # strip inline Jira markup ({code}, {quote}, etc.) and excess whitespace
            text = re.sub(r"\{[^}]+\}", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                parts.append(text[:400])
                matched_any = True

    if not matched_any:
        # Fall back: strip noisy template headers, keep the substantive prose
        cleaned = _TEMPLATE_NOISE_RE.sub("", desc)
        cleaned = re.sub(r"\{[^}]+\}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        parts.append(cleaned[:500])

    return " ".join(parts)


def load_acsd() -> list:
    # Patterns that identify engineering/ops tickets rather than customer support issues
    _noise = re.compile(
        r"\b(backport|deploy dumps?|dump assist|assist with dumps?|"
        r"(?:additional|create|add|new|provision|request)[\w\s]*(?:saas[\w\s]*)?data[\s-]*spaces?|"
        r"QPT release|RCA request|clone of)\b",
        re.IGNORECASE,
    )
    tickets = []
    with open(DATA_DIR / "acsd_tickets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if not t["summary"].strip() or not t["description"].strip():
                continue
            if _noise.search(t["summary"]):
                continue
            tickets.append(t)
    return tickets


def load_articles() -> list:
    """Load all article sources and merge into one deduplicated list.

    Sources (in priority order):
      1. data/kb_github_articles.json  — public Experience League KB (~240 articles)
      2. data/dxskb_articles.json      — DXSKB internal/public articles (~93 articles)
    """
    articles = []

    github_path = DATA_DIR / "kb_github_articles.json"
    if github_path.exists():
        with open(github_path) as f:
            for a in json.load(f):
                articles.append({
                    "key": a["key"],
                    "summary": a["summary"],
                    "description": a.get("body_excerpt", a.get("description", "")),
                    "is_public": True,
                    "source": "github-kb",
                    "el_url": a.get("el_url", ""),
                })
        print(f"  Loaded {len(articles)} public GitHub KB articles")
    else:
        print("  WARNING: kb_github_articles.json not found — run fetch_kb_github.py first")

    dxskb_path = DATA_DIR / "dxskb_articles.json"
    if dxskb_path.exists():
        before = len(articles)
        with open(dxskb_path) as f:
            for a in json.load(f):
                articles.append({
                    "key": a["key"],
                    "summary": a["summary"],
                    "description": a.get("description", ""),
                    "is_public": a.get("is_public", False),
                    "source": "dxskb",
                    "el_url": "",
                })
        print(f"  Loaded {len(articles) - before} DXSKB articles")

    return articles


# keep backward-compat name
def load_dxskb() -> list:
    with open(DATA_DIR / "dxskb_articles.json") as f:
        return json.load(f)


def embed_texts(texts: list, model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def cluster_tickets(embeddings: np.ndarray) -> np.ndarray:
    # HDBSCAN finds clusters of varying density, unlike DBSCAN which uses a
    # single fixed eps radius. With DBSCAN eps=0.25, all tickets sharing a
    # product-area vocabulary (Live Search, API Mesh, etc.) chain into one blob
    # regardless of the actual issue. HDBSCAN's density hierarchy breaks these
    # blobs into tighter, problem-specific clusters.
    #
    # min_cluster_size=5: minimum tickets to form a cluster
    # min_samples=5: how many neighbors a point needs to be a core point —
    #   higher = more conservative, more noise points, tighter clusters
    # metric="euclidean" on normalized embeddings is equivalent to cosine distance
    db = HDBSCAN(min_cluster_size=3, min_samples=3, metric="euclidean", n_jobs=-1)
    return db.fit_predict(embeddings)


def detect_gap(centroid: np.ndarray, article_embeddings: np.ndarray, articles: list):
    """Return (doc_status, best_similarity, top_3_article_keys).

    Three tiers:
      publicly_documented   — match (≥ WEAK_THRESHOLD) with a public article
      internally_documented — match (≥ WEAK_THRESHOLD) with an internal-only article
      gap                   — no meaningful match found (< WEAK_THRESHOLD)

    STRONG_THRESHOLD is used for confidence scoring only — the public/internal
    distinction is driven entirely by is_public on the best-matching article.
    Previously, the WEAK tier always returned "internally_documented" even when
    the nearest article was a public EL article — that was a logic bug.
    """
    sims = article_embeddings @ centroid
    top_idx = np.argsort(sims)[::-1][:3]
    best_sim = float(sims[top_idx[0]])
    top_keys = [articles[i]["key"] for i in top_idx]
    best_article = articles[top_idx[0]]

    if best_sim >= WEAK_THRESHOLD:
        status = "publicly_documented" if best_article["is_public"] else "internally_documented"
    else:
        status = "gap"

    return status, round(best_sim, 3), top_keys


def representative_summary(cluster_tickets: list, cluster_embeddings: np.ndarray) -> str:
    """Pick the ticket summary closest to the cluster centroid."""
    centroid = cluster_embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    best_idx = int(np.argmax(cluster_embeddings @ centroid))
    return cluster_tickets[best_idx]["summary"]


def run():
    print("Loading data...")
    tickets = load_acsd()
    articles = load_articles()
    public_count = sum(1 for a in articles if a["is_public"])
    print(f"  {len(tickets)} ACSD tickets | {len(articles)} KB articles ({public_count} public)")

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("\nEmbedding ACSD tickets (problem-focused text only)...")
    ticket_embeddings = embed_texts(
        [extract_ticket_text(t) for t in tickets], model
    )

    print("\nClustering tickets...")
    labels = cluster_tickets(ticket_embeddings)
    unique_labels = sorted(set(labels) - {-1})
    print(f"  {len(unique_labels)} clusters | {int(sum(labels == -1))} noise tickets")

    print("\nEmbedding KB articles...")
    article_embeddings = embed_texts(
        [f"{a['summary']} {a['description'][:500]}" for a in articles], model
    )

    print("\nDetecting documentation gaps...")
    results = []
    for label in tqdm(unique_labels):
        mask = labels == label
        c_tickets = [t for t, m in zip(tickets, mask) if m]
        c_embeddings = ticket_embeddings[mask]

        centroid = c_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        status, best_sim, top_keys = detect_gap(centroid, article_embeddings, articles)
        summary = representative_summary(c_tickets, c_embeddings)

        # build article metadata for top matches
        article_by_key = {a["key"]: a for a in articles}
        top_articles = [
            {
                "key": k,
                "summary": article_by_key.get(k, {}).get("summary", ""),
                "source": article_by_key.get(k, {}).get("source", ""),
                "el_url": article_by_key.get(k, {}).get("el_url", ""),
                "is_public": article_by_key.get(k, {}).get("is_public", False),
            }
            for k in top_keys
        ]

        results.append({
            "cluster_id": int(label),
            "size": len(c_tickets),
            "summary": summary,
            "doc_status": status,
            "best_similarity": best_sim,
            "top_dxskb_keys": top_keys,          # backward compat for dashboard
            "top_articles": top_articles,          # enriched with source + EL URL
            "ticket_keys": [t["key"] for t in c_tickets],
        })

    results.sort(key=lambda x: -x["size"])

    output = DATA_DIR / "clusters.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    qualifying = [r for r in results if r["size"] >= MIN_CLUSTER_SIZE]
    gaps = sum(1 for r in qualifying if r["doc_status"] == "gap")
    internal = sum(1 for r in qualifying if r["doc_status"] == "internally_documented")
    public = sum(1 for r in qualifying if r["doc_status"] == "publicly_documented")

    print(f"\n{'='*50}")
    print(f"Total clusters          : {len(results)}")
    print(f"Qualifying (≥{MIN_CLUSTER_SIZE} tickets) : {len(qualifying)}")
    print(f"  Documentation gaps    : {gaps}")
    print(f"  Internally documented : {internal}")
    print(f"  Publicly documented   : {public}")
    print(f"\nResults → {output}")


if __name__ == "__main__":
    run()
