# KB Documentation Gap Detection — POC

Detects recurring Adobe Commerce support issues that lack public KB documentation.
Embeds ACSD tickets → clusters by semantic similarity → cross-references against Experience League articles → surfaces gaps in a Streamlit dashboard.

Wiki: https://wiki.corp.adobe.com/spaces/~makwana/pages/3901727440/KB+Documentation+Gap+Detection+%E2%80%94+POC

---

## Prerequisites

- Python 3.9+
- `.env` file in this directory with:
  ```
  JIRA_BASE=https://jira.corp.adobe.com
  JIRA_PAT=<your PAT>
  ```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

### Step 1 — Fetch data (run when you want fresh data)

```bash
python fetch_acsd.py        # ACSD tickets → data/acsd_tickets.jsonl
python fetch_dxskb.py       # DXSKB KB articles → data/dxskb_articles.json
python fetch_kb_github.py   # EL articles from GitHub → data/kb_github_articles.json
```

### Step 2 — Run pipeline

```bash
python pipeline.py          # embed → cluster → gap detect → data/clusters.json
```

### Step 3 — Launch dashboard

```bash
streamlit run dashboard.py --server.headless true
# Open http://localhost:8501
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Thresholds, paths, model name |
| `fetch_acsd.py` | Fetch ACSD tickets from Jira |
| `fetch_dxskb.py` | Fetch DXSKB KB articles from Jira |
| `fetch_kb_github.py` | Clone AdobeDocs repos, parse EL articles |
| `pipeline.py` | Embed → cluster → gap detect |
| `dashboard.py` | Streamlit dashboard |

## Data files (generated, not committed)

| File | Contents |
|---|---|
| `data/acsd_tickets.jsonl` | Raw ACSD tickets |
| `data/dxskb_articles.json` | DXSKB KB articles |
| `data/kb_github_articles.json` | Parsed EL articles from GitHub repos |
| `data/clusters.json` | Pipeline output — clusters + gap status |

---

## Config knobs (`config.py`)

| Setting | Default | Effect |
|---|---|---|
| `MIN_CLUSTER_SIZE` | 3 | Minimum tickets to show in dashboard |
| `WEAK_THRESHOLD` | 0.60 | Cosine similarity cutoff for "documented" |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
