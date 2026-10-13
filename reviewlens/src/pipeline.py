"""End-to-end analysis: collect -> score with the ML model -> extract aspects -> store in SQL."""
from typing import Callable

from src import db
from src.collect import search_expert_reviews, search_offers, split_pasted_reviews
from src.extract import analyze_customers, analyze_expert, choose_aspects
from src.scoring import predict_stars

MAX_CUSTOMER_REVIEWS_FOR_ASPECTS = 120  # the ML model scores every review; the LLM tags up to this many


def run(conn, product: str, pasted: list[dict], use_web: bool = True,
        progress: Callable[[str], None] = print) -> tuple[int, list[str]]:
    """pasted: [{"source": "Amazon", "text": "...raw pasted reviews..."}].
    Returns (search_id, aspects)."""
    search_id = db.new_search(conn, product)

    customer_reviews = []  # (source, text, stated_stars)
    for batch in pasted:
        found = split_pasted_reviews(batch["text"])
        with_stars = sum(r["stated_stars"] is not None for r in found)
        progress(f"{batch['source']}: read {len(found)} reviews ({with_stars} with star ratings)")
        if found and not with_stars and "---" not in batch["text"]:
            progress(f"   tip: no star lines or --- found in {batch['source']}, so each paragraph "
                     "was treated as a separate review")
        for r in found:
            customer_reviews.append((batch["source"], r["text"], r["stated_stars"]))
    if customer_reviews:
        progress(f"Read {len(customer_reviews)} pasted customer reviews")

    articles, aspects = [], None
    cached = db.query(conn, "recent_search", product=product) if use_web else None
    if use_web and cached is not None and len(cached) and int(cached.id[0]) != search_id:
        old = int(cached.id[0])
        db.copy_expert_data(conn, old, search_id)
        aspects = db.get_aspects(conn, old)
        progress("Reused expert reviews from a search in the last 24 hours (cached in SQL)")
    elif use_web:
        progress("Searching the web for expert reviews...")
        articles = search_expert_reviews(product)
        progress(f"Found {len(articles)} candidate review articles")

    if not aspects:
        samples = [a["text"] for a in articles] + [c[1] for c in customer_reviews[:10]]
        if not samples:
            raise ValueError("No reviews found. Try a more specific product name or paste reviews.")
        try:
            aspects = choose_aspects(product, samples)
        except Exception as e:  # noqa: BLE001
            progress(f"   couldn't pick aspects automatically ({type(e).__name__}); using general ones")
            aspects = ["Quality", "Performance", "Ease of use", "Durability", "Price/value"]
    db.set_aspects(conn, search_id, aspects)
    progress(f"Aspects to analyze: {', '.join(aspects)}")

    # ---- Experts: LLM summarizes each article's verdict, the ML model scores that verdict
    kept = 0
    for i, art in enumerate(articles, 1):
        progress(f"Reading expert review {i}/{len(articles)}: {art['source']}")
        try:
            result = analyze_expert(product, aspects, art)
        except Exception as e:  # noqa: BLE001
            progress(f"   skipped ({type(e).__name__})")
            continue
        if result is None:
            progress("   skipped: not a review of this product")
            continue
        stars = float(predict_stars([result["verdict"]])[0])
        rid = db.add_review(conn, search_id, "expert", art["source"], result["verdict"], stars,
                            art["title"], art["url"], result["stated_score"])
        db.add_aspects(conn, rid, result["mentions"])
        kept += 1
    if articles:
        progress(f"Kept {kept} expert reviews")

    # ---- Customers: ML model scores every review; LLM tags aspects on up to N of them
    if customer_reviews:
        texts = [c[1] for c in customer_reviews]
        stars = predict_stars(texts)
        progress(f"Scored {len(texts)} customer reviews with the sentiment model")
        n = min(len(texts), MAX_CUSTOMER_REVIEWS_FOR_ASPECTS)
        progress(f"Finding aspects in {n} customer reviews...")
        try:
            mentions = analyze_customers(product, aspects, texts[:n]) + [[] for _ in texts[n:]]
        except Exception as e:  # noqa: BLE001
            progress(f"   aspect tagging failed ({type(e).__name__}); ratings still available")
            mentions = [[] for _ in texts]
        for (source, text, stated), s, m in zip(customer_reviews, stars, mentions):
            rid = db.add_review(conn, search_id, "customer", source, text, float(s), stated_score=stated)
            db.add_aspects(conn, rid, m)

    # ---- Where to buy
    if use_web and not len(db.query(conn, "offers", search_id=search_id)):
        progress("Checking US store prices...")
        try:
            db.add_offers(conn, search_id, search_offers(product))
        except Exception:  # noqa: BLE001
            progress("   price search failed; skipping")
    conn.commit()
    progress("Done")
    return search_id, aspects