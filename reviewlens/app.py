"""Streamlit app: run with  streamlit run app.py"""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

APP_NAME = "ReviewLens"
ROOT = Path(__file__).parent
SOURCES = ["Amazon", "Best Buy", "Walmart", "Target", "eBay", "Flipkart", "Other"]

load_dotenv(ROOT / ".env")
for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):  # Streamlit Cloud secrets -> environment
    try:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
    except Exception:  # noqa: BLE001  (no secrets file locally is fine)
        pass

from src import db  # noqa: E402
from src.pipeline import run  # noqa: E402

st.set_page_config(page_title=APP_NAME, page_icon="⭐", layout="wide")


@st.cache_resource
def get_conn():
    return db.connect()


conn = get_conn()
st.title(f"⭐ {APP_NAME}")
st.subheader("Expert and customer reviews, analyzed side by side")
st.caption("Type any product. The app finds expert reviews on the web, scores them with a sentiment "
           "model trained on 42K real reviews, and breaks opinions down by aspect, with quotes linked "
           "to their sources. Paste reviews from Amazon, Best Buy, and more to add the customer view.")

# ---------------- Input ----------------
with st.form("analyze"):
    product = st.text_input("Product name", placeholder="e.g. Sony WH-1000XM5 headphones")
    use_web = st.checkbox("Search the web for expert reviews", value=True)
    with st.expander("➕ Add customer reviews (optional): paste from Amazon, Best Buy, Walmart, etc."):
        st.caption("Copy reviews from a store page and paste them below. **Include each review's star line** "
                   "(e.g. '4.0 out of 5 stars Great sound'): it keeps multi-paragraph reviews together and lets "
                   "the app check its model against real ratings. No star lines? Put a line with --- between reviews.")
        pasted = []
        for i in range(3):
            c1, c2 = st.columns([1, 4])
            src = c1.selectbox("Store", SOURCES, index=i, key=f"src{i}")
            txt = c2.text_area("Reviews", key=f"txt{i}", height=120, label_visibility="collapsed",
                               placeholder=f"Paste {SOURCES[i]} reviews here...")
            pasted.append({"source": src, "text": txt})
        c1, c2 = st.columns([1, 4])
        file_src = c1.selectbox("Store", SOURCES, index=6, key="filesrc")
        upload = c2.file_uploader("Or upload a CSV (with a 'review' column) or a .txt file",
                                  type=["csv", "txt"])
    go = st.form_submit_button("Analyze reviews", type="primary")

if go:
    batches = [p for p in pasted if p["text"].strip()]
    if upload is not None:
        if upload.name.endswith(".csv"):
            df = pd.read_csv(upload)
            col = next((c for c in df.columns if c.lower() in ("review", "reviews", "text", "review_text", "body")), None)
            if col is None:
                st.error("CSV needs a column named 'review' or 'text'.")
                st.stop()
            batches.append({"source": file_src, "text": "\n\n".join(df[col].dropna().astype(str))})
        else:
            batches.append({"source": file_src, "text": upload.read().decode("utf-8", "ignore")})
    if not product.strip():
        st.warning("Please type a product name.")
    elif not use_web and not batches:
        st.warning("Turn on web search or paste some reviews.")
    elif use_web and not (os.environ.get("GROQ_API_KEY") and os.environ.get("TAVILY_API_KEY")):
        st.error("Missing API keys. Add GROQ_API_KEY and TAVILY_API_KEY to your .env file or Streamlit secrets.")
    else:
        with st.status("Analyzing reviews...", expanded=True) as status:
            try:
                sid, aspects = run(conn, product.strip(), batches, use_web=use_web, progress=st.write)
                st.session_state["search"] = {"id": sid, "product": product.strip()}
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                status.update(label="Analysis failed", state="error")
                st.error(str(e))

# ---------------- Results ----------------
if "search" not in st.session_state:
    st.stop()

sid, product = st.session_state["search"]["id"], st.session_state["search"]["product"]
by_type = db.query(conn, "rating_by_source_type", search_id=sid).set_index("source_type")
if by_type.empty:
    st.warning("No reviews could be analyzed. Try a more specific product name, or paste customer reviews.")
    st.stop()

st.header(f"Results for: {product}")
stars = lambda x: "★" * int(round(x)) + "☆" * (5 - int(round(x)))  # noqa: E731
c1, c2 = st.columns(2)
for col, kind, label in ((c1, "expert", "Expert rating"), (c2, "customer", "Customer rating")):
    if kind in by_type.index:
        r = by_type.loc[kind]
        col.metric(label, f"{r.avg_stars:.1f} / 5   {stars(r.avg_stars)}", f"{int(r.reviews)} review{'' if int(r.reviews) == 1 else 's'}", delta_color="off")
        if pd.notna(r.avg_stated_score):
            who = "reviewers" if kind == "expert" else "customers"
            col.caption(f"Scores the {who} gave themselves average {r.avg_stated_score:.1f} / 5")
    else:
        col.metric(label, "—", "none yet", delta_color="off")
        if kind == "customer":
            col.caption("Paste reviews from a store to add customer opinions.")
st.caption("Expert and customer ratings are shown separately on purpose: experts test a product for days, "
           "customers live with it for months, and they often notice different things.")

tab_aspects, tab_buy, tab_sources, tab_reviews, tab_model = st.tabs(
    ["🧩 Aspects, pros & cons", "🛒 Where to buy", "📊 By source", "📝 All reviews", "🤖 About the model"])

# ---- Aspects, pros & cons
with tab_aspects:
    asp = db.query(conn, "aspect_scores", search_id=sid)
    quotes = db.query(conn, "top_quotes", search_id=sid)
    if asp.empty:
        st.info("No aspect mentions found.")
    else:
        asp["who"] = asp.source_type.map({"expert": "Experts", "customer": "Customers"})
        asp["start"] = 1.0
        st.altair_chart(
            alt.Chart(asp).mark_bar().encode(
                y=alt.Y("aspect:N", title=None, axis=alt.Axis(labelLimit=250)),
                x=alt.X("start:Q", title="Aspect score (1 = mostly negative, 5 = mostly positive)",
                        scale=alt.Scale(domain=[1, 5])),
                x2="score:Q",
                color=alt.Color("who:N", title=None),
                yOffset="who:N",
                tooltip=["who", "aspect", "score", "mentions", "positive", "negative"],
            ).properties(height=60 * asp.aspect.nunique() + 40, title="Aspect scores: experts vs customers"),
            width="stretch")

        overall = asp.groupby("aspect")[["mentions", "positive", "negative"]].sum()
        overall["score"] = 3 + 2 * (overall.positive - overall.negative) / overall.mentions
        pros = overall[overall.score >= 3.5].sort_values("score", ascending=False)
        cons = overall[overall.score <= 2.5].sort_values("score")
        mixed = overall[(overall.score > 2.5) & (overall.score < 3.5)]

        used = set()

        def show(aspect, sentiment):
            q = quotes[(quotes.aspect == aspect) & (quotes.sentiment == sentiment)
                       & ~quotes.quote.isin(used)]
            if len(q):
                q = q.iloc[0]
                used.add(q.quote)
                link = f"[{q.source}]({q.url})" if isinstance(q.url, str) and q.url else q.source
                st.markdown(f"> “{q.quote}”  \n> — {link}")

        p, c = st.columns(2)
        with p:
            st.markdown("### 👍 Pros")
            for a, r in pros.iterrows():
                st.markdown(f"**{a}** · {int(r.positive)} positive / {int(r.negative)} negative mentions")
                show(a, "positive")
            if pros.empty:
                st.caption("No clearly positive aspects.")
        with c:
            st.markdown("### 👎 Cons")
            for a, r in cons.iterrows():
                st.markdown(f"**{a}** · {int(r.negative)} negative / {int(r.positive)} positive mentions")
                show(a, "negative")
            if cons.empty:
                st.caption("No clearly negative aspects.")
        if not mixed.empty:
            st.markdown("### ⚖️ Mixed opinions")
            for a, r in mixed.iterrows():
                st.markdown(f"**{a}** · {int(r.positive)} positive / {int(r.negative)} negative mentions")
                show(a, "positive")
                show(a, "negative")

# ---- Where to buy
with tab_buy:
    offers = db.query(conn, "offers", search_id=sid)
    if offers.empty:
        st.info("No US store listings found in search results.")
    else:
        st.dataframe(offers, hide_index=True, width="stretch", column_config={
            "store": "Store", "title": "Listing",
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
            "price": st.column_config.NumberColumn("Price seen", format="$%.2f")})
        st.caption("Prices are taken from search results and may be out of date or refer to a "
                   "different variant. Always check the store for the current price.")

# ---- By source
with tab_sources:
    src = db.query(conn, "rating_by_source", search_id=sid)
    src["Type"] = src.source_type.str.title()
    st.dataframe(src[["Type", "source", "avg_stars", "reviews"]], hide_index=True, width="stretch",
                 column_config={"source": "Source", "avg_stars": st.column_config.NumberColumn("Model rating", format="%.2f ★"),
                                "reviews": "Reviews"})
    dist = db.query(conn, "star_distribution", search_id=sid)
    dist["who"] = dist.source_type.map({"expert": "Experts", "customer": "Customers"})
    st.altair_chart(alt.Chart(dist).mark_bar().encode(
        x=alt.X("star:O", title="Predicted stars"), y=alt.Y("reviews:Q", title="Reviews"),
        color=alt.Color("who:N", title=None), xOffset="who:N",
    ).properties(title="Rating distribution", height=280), width="stretch")

# ---- All reviews
with tab_reviews:
    rev = pd.read_sql_query("SELECT source_type, source, ROUND(stars, 2) AS stars, stated_score, text, url "
                            "FROM reviews WHERE search_id = ? ORDER BY source_type, stars DESC", conn, params=(sid,))
    st.dataframe(rev, hide_index=True, width="stretch", column_config={
        "source_type": "Type", "source": "Source",
        "stars": st.column_config.NumberColumn("Model rating", format="%.2f"),
        "stated_score": st.column_config.NumberColumn("Their own rating", format="%.1f"),
        "text": st.column_config.TextColumn("Review (expert reviews show the summarized verdict)", width="large"),
        "url": st.column_config.LinkColumn("Link", display_text="Open")})
    checked = rev.dropna(subset=["stated_score"])
    if len(checked) >= 3:
        err = (checked.stars - checked.stated_score).abs()
        st.success(f"Model check: on {len(checked)} reviews where the author gave their own rating, the model's "
                   f"prediction was off by {err.mean():.2f} stars on average, and within 1 star "
                   f"{(err <= 1).mean():.0%} of the time.")

# ---- About the model
with tab_model:
    m = json.loads((ROOT / "models" / "sentiment_metrics.json").read_text())
    rs = m["random_split"]
    st.markdown(f"""
The star ratings come from a **TF-IDF + Logistic Regression model** trained on **{m['training_size']:,} real
product reviews** from three very different categories (food, music & audio gear, clothing). It predicts the
probability of each star rating, and the app reports the expected value (e.g. 4.3 stars).

| Test | Exact star | Within 1 star | Positive / neutral / negative correct | Avg error |
|---|---|---|---|---|
| Random 20% hold-out | {rs['exact_star_accuracy']:.0%} | {rs['within_one_star']:.0%} | {rs['polarity_accuracy']:.0%} | {rs['mean_abs_error_stars']:.2f} ★ |
""" + "\n".join(
        f"| Unseen category: {k.replace('_', ' & ')} | {v['exact_star_accuracy']:.0%} | {v['within_one_star']:.0%} | "
        f"{v['polarity_accuracy']:.0%} | {v['mean_abs_error_stars']:.2f} ★ |" for k, v in m["unseen_category"].items()) + """

The *unseen category* rows train on two categories and test on the third, showing how the model handles
product types it never saw. An LLM (Groq) is used only to summarize expert verdicts and to tag aspects;
every quote shown is verified to appear word-for-word in its source.
""")