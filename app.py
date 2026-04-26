"""Streamlit UI for the AI Market Research assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

import storage
from pipeline import run_daily
from scheduler import DailyScheduler

load_dotenv()

st.set_page_config(
    page_title="AI Market Research — Virtual Crew",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Boot the background scheduler exactly once per process --------------
@st.cache_resource
def get_scheduler() -> DailyScheduler:
    s = DailyScheduler()
    s.start()
    return s


scheduler = get_scheduler()
cfg = storage.load_config()


# ---- Sidebar: settings ---------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("`ANTHROPIC_API_KEY` is not set. Add it to your environment or `.env`.")

    with st.expander("🕒 Delivery schedule", expanded=True):
        cfg["delivery_time"] = st.text_input(
            "Daily run time (HH:MM, 24h, local)", value=cfg.get("delivery_time", "07:30")
        )
        cfg["timezone"] = st.text_input("Timezone label", value=cfg.get("timezone", "Asia/Kolkata"))
        cfg["auto_run_in_app"] = st.checkbox(
            "Auto-generate when app opens (if today's brief is missing)",
            value=cfg.get("auto_run_in_app", True),
        )
        if scheduler.next_run:
            st.caption(f"Next scheduled run: **{scheduler.next_run:%Y-%m-%d %H:%M}**")
        if scheduler.last_run:
            st.caption(f"Last run: {scheduler.last_run:%Y-%m-%d %H:%M}")
        if scheduler.last_error:
            st.warning(f"Last error: {scheduler.last_error}")

    with st.expander("🏭 Industry focus", expanded=True):
        st.caption("Leave empty for sector-agnostic. Add any vertical you care about.")
        preset = [
            "Healthcare", "Finance", "Retail", "Manufacturing", "Defense",
            "Education", "Media", "Legal", "Energy", "Logistics",
            "Public Sector", "Developer Tools", "Consumer", "Enterprise SaaS",
            "Hardware/Chips", "Research/Academia", "Pharma", "Insurance", "Real Estate",
        ]
        cfg["industries"] = st.multiselect(
            "Industries to weight",
            options=sorted(set(preset + cfg.get("industries", []))),
            default=cfg.get("industries", []),
        )
        custom = st.text_input("Add custom industry (comma-separated)", "")
        if custom.strip():
            extra = [s.strip() for s in custom.split(",") if s.strip()]
            cfg["industries"] = sorted(set(cfg["industries"] + extra))

    with st.expander("📡 Sources"):
        cfg["enabled_categories"] = st.multiselect(
            "Feed categories",
            options=["press", "corp_research", "business", "research"],
            default=cfg.get("enabled_categories", ["press", "corp_research", "business", "research"]),
        )
        cfg["include_hackernews"] = st.checkbox(
            "Include Hacker News", value=cfg.get("include_hackernews", True)
        )
        cfg["include_reddit"] = st.checkbox(
            "Include Reddit", value=cfg.get("include_reddit", True)
        )
        cfg["newsapi_query"] = st.text_input(
            "Optional NewsAPI query (requires NEWSAPI_KEY env var)",
            value=cfg.get("newsapi_query", ""),
        )

    with st.expander("🤖 Model"):
        cfg["model"] = st.selectbox(
            "Claude model",
            options=["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
            index=["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"].index(
                cfg.get("model", "claude-sonnet-4-6")
            ),
        )
        cfg["max_articles"] = st.slider(
            "Max articles fed to Claude", 30, 150, cfg.get("max_articles", 80), step=10
        )

    with st.expander("✉️ Email delivery"):
        em = cfg.get("email", {})
        em["enabled"] = st.checkbox("Email me the daily briefing", value=em.get("enabled", False))
        em["to"] = st.text_input("To address", value=em.get("to", ""))
        em["from"] = st.text_input("From address (optional)", value=em.get("from", ""))
        em["smtp_host"] = st.text_input("SMTP host", value=em.get("smtp_host", "smtp.gmail.com"))
        em["smtp_port"] = st.number_input("SMTP port", value=int(em.get("smtp_port", 587)))
        em["smtp_user"] = st.text_input("SMTP username", value=em.get("smtp_user", ""))
        em["smtp_password"] = st.text_input(
            "SMTP password / app password", value=em.get("smtp_password", ""), type="password"
        )
        cfg["email"] = em

    if st.button("💾 Save settings", use_container_width=True):
        storage.save_config(cfg)
        st.success("Saved.")
        st.rerun()

    st.divider()
    if st.button("🚀 Run briefing now", use_container_width=True, type="primary"):
        with st.spinner("Fetching sources and analyzing with Claude…"):
            try:
                run_daily(cfg)
                st.success("Briefing ready.")
            except Exception as e:
                st.error(f"Run failed: {e}")
        st.rerun()

    st.divider()
    st.markdown("### 📅 History")
    dates = storage.list_dates(30)
    if dates:
        chosen = st.selectbox("Open a past briefing", options=dates)
    else:
        chosen = None
        st.caption("No past briefings yet.")


# ---- Pick which briefing to render ---------------------------------------
def pick_briefing():
    if chosen:
        b = storage.briefing_for_date(chosen)
        if b:
            return b, chosen
    b = storage.latest_briefing()
    if b:
        return b, b.get("_meta", {}).get("generated_at", "")[:10]
    return None, None


briefing, brief_date = pick_briefing()

# Auto-run on first open if no briefing for today and user opted in
today = datetime.utcnow().date().isoformat()
if cfg.get("auto_run_in_app") and (not briefing or brief_date != today) and os.getenv("ANTHROPIC_API_KEY"):
    if "auto_run_done" not in st.session_state:
        st.session_state["auto_run_done"] = True
        with st.spinner("First time today — generating your briefing…"):
            try:
                run_daily(cfg)
            except Exception as e:
                st.warning(f"Auto-run failed: {e}")
        briefing, brief_date = pick_briefing()


# ---- Header --------------------------------------------------------------
st.markdown("# 🧠 AI Market Research — Virtual Crew")
st.caption(
    "Daily executive briefing on the AI industry: top stories, leader voices, "
    "big-tech moves, and *what companies are actually shipping*."
)

if not briefing:
    st.info(
        "No briefing yet. Configure your settings in the sidebar and click "
        "**🚀 Run briefing now** to generate the first one."
    )
    st.stop()


meta = briefing.get("_meta", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Briefing date", brief_date or "—")
c2.metric("Articles analyzed", meta.get("article_count", 0))
c3.metric("Stories surfaced", len(briefing.get("top_stories", [])))
c4.metric("Implementations tracked", len(briefing.get("implementations", [])))


# ---- Tabs ----------------------------------------------------------------
tab_brief, tab_leaders, tab_bigtech, tab_impl, tab_charts, tab_raw = st.tabs(
    ["📰 Daily Briefing", "🎙️ Industry Leaders", "🏢 Big Tech", "🛠 Implementations", "📊 Analysis", "🔬 Raw"]
)


# --- Daily Briefing -------------------------------------------------------
with tab_brief:
    st.markdown(f"### {briefing.get('headline','')}")
    st.write(briefing.get("executive_summary", ""))
    st.markdown("#### Top stories")
    for s in briefing.get("top_stories", []):
        with st.container(border=True):
            st.markdown(f"**[{s.get('title','')}]({s.get('url','')})** — *{s.get('source','')}*")
            badge_cols = st.columns([1, 1, 4])
            badge_cols[0].caption(f"🏭 {s.get('industry','—')}")
            badge_cols[1].caption(f"📈 {s.get('stage','—')}")
            if s.get("companies"):
                badge_cols[2].caption("🏷 " + ", ".join(s["companies"]))
            st.write(s.get("why_it_matters", ""))

    if briefing.get("themes"):
        st.markdown("#### Today's themes")
        st.write(" · ".join(f"**{t}**" for t in briefing["themes"]))
    if briefing.get("watchlist"):
        st.markdown("#### 👀 Watchlist")
        for w in briefing["watchlist"]:
            st.markdown(f"- {w}")


# --- Industry leaders -----------------------------------------------------
with tab_leaders:
    leaders = briefing.get("industry_leaders", [])
    if not leaders:
        st.info("No leader quotes captured today.")
    for l in leaders:
        with st.container(border=True):
            st.markdown(f"**{l.get('person','')}** — *{l.get('role','')}*")
            st.write(f"> {l.get('quote_or_position','')}")
            if l.get("url"):
                st.markdown(f"[Source]({l['url']})")


# --- Big Tech -------------------------------------------------------------
with tab_bigtech:
    moves = briefing.get("big_tech_moves", [])
    if not moves:
        st.info("No big-tech moves captured today.")
    df_moves = pd.DataFrame(moves)
    if not df_moves.empty:
        st.dataframe(df_moves, use_container_width=True, hide_index=True)
    for m in moves:
        with st.container(border=True):
            st.markdown(f"**{m.get('company','')}** — {m.get('move','')}")
            st.caption(f"Signal: {m.get('signal','')}")
            if m.get("url"):
                st.markdown(f"[Source]({m['url']})")


# --- Implementations ------------------------------------------------------
with tab_impl:
    impls = briefing.get("implementations", [])
    if not impls:
        st.info("No concrete implementations surfaced today.")
    else:
        df = pd.DataFrame(impls)
        st.dataframe(
            df[["company", "industry", "stage", "what", "outcome_or_metric", "url"]]
            if set(["company", "industry", "stage", "what"]).issubset(df.columns)
            else df,
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("source")},
        )


# --- Analysis charts ------------------------------------------------------
with tab_charts:
    impls = briefing.get("implementations", [])
    stories = briefing.get("top_stories", [])
    moves = briefing.get("big_tech_moves", [])

    col1, col2 = st.columns(2)
    if impls:
        df = pd.DataFrame(impls)
        with col1:
            st.markdown("#### Implementations by industry")
            counts = df["industry"].fillna("Other").value_counts().reset_index()
            counts.columns = ["industry", "count"]
            st.plotly_chart(
                px.bar(counts, x="industry", y="count", color="industry"),
                use_container_width=True,
            )
        with col2:
            st.markdown("#### Stage distribution")
            stage = df["stage"].fillna("unknown").value_counts().reset_index()
            stage.columns = ["stage", "count"]
            st.plotly_chart(
                px.pie(stage, names="stage", values="count", hole=0.4),
                use_container_width=True,
            )

    company_mentions = Counter()
    for s in stories:
        for c in s.get("companies", []) or []:
            company_mentions[c] += 1
    for m in moves:
        if m.get("company"):
            company_mentions[m["company"]] += 1
    if company_mentions:
        st.markdown("#### Company mention share")
        df_c = pd.DataFrame(company_mentions.most_common(15), columns=["company", "mentions"])
        st.plotly_chart(
            px.bar(df_c, x="mentions", y="company", orientation="h").update_layout(
                yaxis={"categoryorder": "total ascending"}
            ),
            use_container_width=True,
        )

    # 30-day briefing volume
    dates = storage.list_dates(30)
    if len(dates) > 1:
        st.markdown("#### Briefings over the last 30 days")
        df_d = pd.DataFrame({"date": dates, "count": 1})
        st.plotly_chart(px.bar(df_d, x="date", y="count"), use_container_width=True)


# --- Raw payload ----------------------------------------------------------
with tab_raw:
    st.json(briefing)
