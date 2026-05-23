import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

JIRA_BASE = os.environ["JIRA_BASE"]
JIRA_PAT = os.environ["JIRA_PAT"]

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Minimum tickets in a cluster to qualify as recurring
MIN_CLUSTER_SIZE = 3

# Cosine similarity thresholds for gap detection
# Above STRONG_THRESHOLD  → Publicly documented
# Between WEAK and STRONG → Internally documented only
# Below WEAK_THRESHOLD    → Documentation gap
STRONG_THRESHOLD = 0.75
WEAK_THRESHOLD = 0.60

EMBED_MODEL = "all-MiniLM-L6-v2"
