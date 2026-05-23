"""KB Documentation Gap Dashboard — Streamlit app."""
import json
import pandas as pd
import streamlit as st
from config import DATA_DIR, MIN_CLUSTER_SIZE

st.set_page_config(
    page_title="KB Gap Dashboard",
    page_icon="📊",
    layout="wide",
)

STATUS_LABELS = {
    "gap": "Documentation Gap",
    "internally_documented": "Internally Documented",
    "publicly_documented": "Publicly Documented",
}

STATUS_COLORS = {
    "gap": "🔴",
    "internally_documented": "🟡",
    "publicly_documented": "🟢",
}


@st.cache_data
def load_data():
    clusters_path = DATA_DIR / "clusters.json"
    dxskb_path = DATA_DIR / "dxskb_articles.json"
    if not clusters_path.exists():
        return None, {}
    with open(clusters_path) as f:
        clusters = json.load(f)
    dxskb = {}
    if dxskb_path.exists():
        with open(dxskb_path) as f:
            for a in json.load(f):
                dxskb[a["key"]] = a
    return clusters, dxskb


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 KB Documentation Gap Dashboard")
st.caption(
    "Recurring Adobe Commerce support issues detected from ACSD tickets, "
    "cross-referenced against DXSKB public articles."
)

clusters, dxskb = load_data()
if clusters is None:
    st.error("No cluster data found. Run `python pipeline.py` first.")
    st.stop()

qualifying = [c for c in clusters if c["size"] >= MIN_CLUSTER_SIZE]

# ── Metrics row ───────────────────────────────────────────────────────────────
gaps = [c for c in qualifying if c["doc_status"] == "gap"]
internal = [c for c in qualifying if c["doc_status"] == "internally_documented"]
public = [c for c in qualifying if c["doc_status"] == "publicly_documented"]
total_tickets = sum(c["size"] for c in qualifying)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Recurring Issue Patterns", len(qualifying))
col2.metric("Total Tickets Covered", total_tickets)
col3.metric("🔴 Documentation Gaps", len(gaps))
col4.metric("🟡 Internally Documented", len(internal))
col5.metric("🟢 Publicly Documented", len(public))

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    selected_statuses = st.multiselect(
        "Filter by status",
        options=["gap", "internally_documented", "publicly_documented"],
        default=["gap", "internally_documented"],
        format_func=lambda x: f"{STATUS_COLORS[x]} {STATUS_LABELS[x]}",
    )
with col_f2:
    min_tickets = st.slider("Minimum ticket count", min_value=3, max_value=50, value=3)

filtered = [
    c for c in qualifying
    if c["doc_status"] in selected_statuses and c["size"] >= min_tickets
]
filtered.sort(key=lambda x: -x["size"])

st.subheader(f"Showing {len(filtered)} issue patterns")

# ── Table ─────────────────────────────────────────────────────────────────────
rows = []
for c in filtered:
    rows.append({
        "Status": f"{STATUS_COLORS[c['doc_status']]} {STATUS_LABELS[c['doc_status']]}",
        "Occurrences": c["size"],
        "Issue Pattern": c["summary"],
        "Similarity Score": f"{c['best_similarity']:.0%}",
        "Nearest Article": ", ".join(
            a.get("summary", a["key"])[:50]
            for a in c.get("top_articles", [])[:2]
        ),
    })

df = pd.DataFrame(rows)
selection = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
)

# ── Drill-down ────────────────────────────────────────────────────────────────
if selection["selection"]["rows"]:
    idx = selection["selection"]["rows"][0]
    cluster = filtered[idx]

    st.divider()
    st.subheader(
        f"{STATUS_COLORS[cluster['doc_status']]} Cluster Detail — "
        f"{cluster['size']} tickets"
    )

    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"**Issue Pattern:** {cluster['summary']}")
        st.markdown(
            f"**Status:** {STATUS_LABELS[cluster['doc_status']]}  |  "
            f"**Best similarity score:** {cluster['best_similarity']:.1%}"
        )

        st.markdown("**Tickets in this cluster:**")
        ticket_keys = cluster["ticket_keys"]
        for key in ticket_keys[:15]:
            st.markdown(
                f"- [{key}](https://jira.corp.adobe.com/browse/{key})"
            )
        if len(ticket_keys) > 15:
            st.caption(f"… and {len(ticket_keys) - 15} more tickets")

    with right:
        st.markdown("**Nearest KB articles:**")
        top_articles = cluster.get("top_articles") or []
        for i, art in enumerate(top_articles):
            key = art["key"]
            el_url = art.get("el_url", "")
            source = art.get("source", "")
            is_pub = art.get("is_public", False)

            if el_url and source == "github-kb":
                pub_badge = "🟢 Public (Experience League)"
                link = f"[{key}]({el_url})"
            elif source == "dxskb":
                pub_badge = "🟢 Public (DXSKB)" if is_pub else "🟡 Internal (DXSKB)"
                link = f"[{key}](https://jira.corp.adobe.com/browse/{key})"
            else:
                pub_badge = "🟢 Public" if is_pub else "🟡 Internal"
                link = f"[{key}](https://jira.corp.adobe.com/browse/{key})"

            title = art.get("summary") or dxskb.get(key, {}).get("summary", "")
            st.markdown(f"- {link} {pub_badge}")
            if title:
                st.caption(f"  {title[:100]}")

        st.markdown(f"**Similarity to nearest article:** {cluster['best_similarity']:.1%}")
        if cluster["doc_status"] == "gap":
            st.error(
                "No public KB article found for this recurring issue. "
                "Consider creating one via Oasis."
            )
        elif cluster["doc_status"] == "internally_documented":
            st.warning(
                "An internal article exists but is not publicly available. "
                "Consider publishing it to Experience League."
            )
        else:
            st.success("This issue is covered by a public Experience League article.")
